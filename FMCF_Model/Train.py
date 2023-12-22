
import tensorflow as tf
from FMCF_Model.FMCF import FMCF
from Decoder.Learning_Rate import Learning_Rate
from Encoder.Built_Transformer import Built_Transformer as utils
import time
from DataOutput import data_prepare, MAX_LENGTH_COMM
from Data_Split_Process.Split_comments import get_dicts
from FMCF_Model.EvaluationMetrics import EvaluationMetrics
import numpy as np
from FMCF_Model.Configs import Train_args
import os

class Train:
    def __init__(self, epochs, sub_data_folder, max_keep, patience, transformer_args):
        self.epochs = epochs
        self.sub_data_folder = sub_data_folder
        self.max_keep = max_keep
        self.patience = patience
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.transformer = FMCF(**transformer_args)
        self.learning_rate = Learning_Rate(transformer_args['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    signature_train = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        ]
    @tf.function(input_signature=signature_train)
    def train_step(self, sbts_train, nodes_train, edges_train, comms_train):
        tar_inp = comms_train[:, :-1]
        tar_real = comms_train[:, 1:]
        sbt_padding_mask, node_padding_mask, look_ahead_mask = utils.create_masks(sbts_train, nodes_train, tar_inp)

        with tf.GradientTape() as tape:
            predictions, attention_weights, _, _ = self.transformer(sbts_train, nodes_train, edges_train, tar_inp, True,
                         sbt_padding_mask, node_padding_mask, look_ahead_mask)
            loss = utils.loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def val_step(self, sbts_val, nodes_val, edges_val, comms_val, val_bleu):
        current_batch_size = comms_val.shape[0]
        tar_reals = comms_val[:, 1:]
        _, _, comms_dic = get_dicts(self.sub_data_folder)
        # add start tag as the first input for the decoder
        decoder_input = [len(comms_dic.word_index)+1] * current_batch_size
        outputs = tf.expand_dims(decoder_input, 1)
        for i in range(MAX_LENGTH_COMM-1):
            sbt_padding_mask, node_padding_mask, look_ahead_mask = utils.create_masks(sbts_val, nodes_val, outputs)
            predictions, attention_weights, _, _ = self.transformer(sbts_val, nodes_val, edges_val, outputs, False,
                         sbt_padding_mask, node_padding_mask, look_ahead_mask)

            predictions = predictions[:, -1:, :]
            predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            outputs = tf.concat([outputs, predicted_ids], axis=-1)
            count_stopped = 0
            for output in outputs:
                if len(comms_dic.word_index) + 2 in output.numpy().tolist():
                    count_stopped += 1
            if count_stopped == current_batch_size:
                break
        candidates = EvaluationMetrics.remove_pad(outputs.numpy().tolist(), len(comms_dic.word_index)+2, "candidates")
        refs = EvaluationMetrics.remove_pad(tar_reals.numpy().tolist(), len(comms_dic.word_index)+2, "references")
        for ref, candi in zip(refs, candidates):
            val_bleu.append(EvaluationMetrics.smoothing1_sentence_bleu(ref, candi))

    def evaluate(self, bleu_avg_best, ckpt_manager, epoch, val_set):
        val_bleu = []
        #val_meteor=[]
        print("Validating ...")

        for batch, (sbts_val, nodes_val, edges_val, comms_val) in enumerate(val_set):
            self.val_step(sbts_val, nodes_val, edges_val, comms_val, val_bleu)
            if batch % 50 == 0:
                print("Validating the {} batch, with val_bleu: {}".format(batch, np.mean(val_bleu)))
                #print("Validating the {} batch, with val_meteor: {}".format(batch, np.mean(val_meteor)))
        bleu_avg_tmp = np.mean(val_bleu)
        print("Validation Completed! average bleu score: {:.4f}".format(bleu_avg_tmp))
        if bleu_avg_tmp > bleu_avg_best:
            bleu_avg_best = bleu_avg_tmp.copy()
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            print("bleu_avg_best: ", bleu_avg_best)
        return bleu_avg_best

    def early_stopping(self, bleu_avg_bests, bleu_avg_best):
        # early stopping
        bleu_avg_bests.append(bleu_avg_best)
        print("bleu_avg_bests: ", bleu_avg_bests)
        if len(bleu_avg_bests) >= self.patience:
            if len(set(bleu_avg_bests[-self.patience:])) == 1:
                print("final bleu_avg_best: ", bleu_avg_best)
                return True

    def train(self):
        train_set, val_set = data_prepare(self.sub_data_folder)
        checkpoint_path = "./checkpoints/" + self.sub_data_folder
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=self.max_keep)
        # 如果检查点存在，则恢复最新的检查点。
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        bleu_avg_best = 0
        bleu_avg_bests = []
        stop_now = False
        for epoch in range(self.epochs):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (sbts_train, nodes_train, edges_train, comms_train)) in enumerate(train_set):
                self.train_step(sbts_train, nodes_train, edges_train, comms_train)
                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                      epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

                # evaluate every 500 batches
                if batch % 500 == 0 and batch != 0:
                    bleu_avg_best = self.evaluate(bleu_avg_best, ckpt_manager, epoch, val_set)
                    if self.early_stopping(bleu_avg_bests, bleu_avg_best) is True:
                        stop_now = True
                        break

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                    self.train_loss.result(), self.train_accuracy.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
            if stop_now is True:
                break
            # Final evaluate
            bleu_avg_best = self.evaluate(bleu_avg_best, ckpt_manager, epoch, val_set)
            if self.early_stopping(bleu_avg_bests, bleu_avg_best) is True:
                break


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train = Train(**Train_args)
    train.train()
"""
这段代码是一个使用TensorFlow进行训练的实现。它定义了一个Train类，其中包含了训练过程中所需的各种方法和属性。

该代码主要的功能如下：
1. 导入必要的模块和库。
2. 定义Train类，包括初始化方法和训练方法。
3. 初始化方法中设置了一些训练参数和模型对象。
4. 训练方法中包含了训练循环，对每个batch进行模型的训练和评估，并在达到一定条件时保存模型。
5. 代码的最后，设置了环境变量并实例化Train类进行训练。

总体而言，这段代码是一个完整的训练流程，使用了TensorFlow的API和自定义模块来构建和训练一个深度学习模型。
"""
