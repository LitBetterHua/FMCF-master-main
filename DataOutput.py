import os
import tensorflow as tf
import numpy as np
import pickle as pkl
import random

BUFFER_SIZE = 20000
# TODO: if you're going to show the generated comments one by one, change the batch_size to 1
# TODO: in normal evaluation, the batch_size = 100
BATCH_SIZE = 100
MAX_LENGTH_SBT = 602
MAX_LENGTH_NODE = 200
MAX_LENGTH_COMM = 22

def cut_if_longer(sbt_seq, node, edge, comm):
  if tf.size(sbt_seq) > MAX_LENGTH_SBT: # only slice if longer
     sbt_seq = tf.slice(sbt_seq, begin=[0], size=[MAX_LENGTH_SBT])
  if tf.size(node) > MAX_LENGTH_NODE:
      node = tf.slice(node, begin=[0], size=[MAX_LENGTH_NODE])
      edge = tf.slice(edge, begin=[0, 0], size = [MAX_LENGTH_NODE, MAX_LENGTH_NODE])
  return sbt_seq, node, edge, comm

def prepare_train_val_test(sbts_train, nodes_train, edges_train, comms_train,
                           sbts_val, nodes_val, edges_val, comms_val):

    def generator_train():
        for sbt, node, edge, comm in zip(sbts_train, nodes_train, edges_train, comms_train):
            yield (sbt, node, edge.todense(), comm)

    def generator_val():
        for sbt, node, edge, comm in zip(sbts_val, nodes_val, edges_val, comms_val):
            yield (sbt, node, edge.todense(), comm)


    train_set = tf.data.Dataset.from_generator(generator=generator_train, output_types=(tf.int32, tf.int32, tf.int32, tf.int32))
    train_set = train_set.map(cut_if_longer)
    train_set = train_set.cache()
    train_set = train_set.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=((None,), (None,), (None, None), (None,)))
    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)


    val_set = tf.data.Dataset.from_generator(generator=generator_val, output_types=(tf.int32,tf.int32, tf.int32, tf.int32))
    val_set = val_set.map(cut_if_longer)
    val_set = val_set.cache()
    val_set = val_set.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=((None,), (None,), (None, None), (None,)))
    val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE)


    return train_set, val_set

def data_prepare(sub_data_folder):
    with open("./datasets/{}/tokens_idx/sbts_train.pkl".format(sub_data_folder), "rb") as fr:
        srcs_train = pkl.load(fr)
    with open("./datasets/{}/tokens_idx/sbts_val.pkl".format(sub_data_folder), "rb") as fr:
        srcs_val = pkl.load(fr)

    with open("./datasets/{}/tokens_idx/nodes_train.pkl".format(sub_data_folder), "rb") as fr:
        nodes_train = pkl.load(fr)
    with open("./datasets/{}/tokens_idx/nodes_val.pkl".format(sub_data_folder), "rb") as fr:
        nodes_val = pkl.load(fr)

    with open("./datasets/{}/tokens_idx/edges_train.pkl".format(sub_data_folder), "rb") as fr:
        edges_train = pkl.load(fr)
    with open("./datasets/{}/tokens_idx/edges_val.pkl".format(sub_data_folder), "rb") as fr:
        edges_val = pkl.load(fr)

    with open("./datasets/{}/tokens_idx/comms_train.pkl".format(sub_data_folder), "rb") as fr:
        comms_train = pkl.load(fr)
    with open("./datasets/{}/tokens_idx/comms_val.pkl".format(sub_data_folder), "rb") as fr:
        comms_val = pkl.load(fr)

    return prepare_train_val_test(srcs_train, nodes_train, edges_train, comms_train,
                           srcs_val, nodes_val, edges_val, comms_val)

def prepare_test(srcs_test, nodes_test, edges_test, comms_test):

    def generator_test():
        for src, node, edge, comm in zip(srcs_test, nodes_test, edges_test, comms_test):
            yield (src, node, edge.todense(), comm)

    test_set = tf.data.Dataset.from_generator(generator=generator_test, output_types=(tf.int32, tf.int32, tf.int32, tf.int32))
    test_set = test_set.map(cut_if_longer)
    test_set = test_set.cache()
    test_set = test_set.padded_batch(BATCH_SIZE, padded_shapes=((None,), (None,), (None, None), (None,)))
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    return test_set

def test_data_prepare(sub_data_folder):
    with open("./datasets/{}/tokens_idx/sbts_test.pkl".format(sub_data_folder), "rb") as fr:
        srcs_test = pkl.load(fr)

    with open("./datasets/{}/tokens_idx/nodes_test.pkl".format(sub_data_folder), "rb") as fr:
        nodes_test = pkl.load(fr)

    with open("./datasets/{}/tokens_idx/edges_test.pkl".format(sub_data_folder), "rb") as fr:
        edges_test = pkl.load(fr)

    with open("./datasets/{}/tokens_idx/comms_test.pkl".format(sub_data_folder), "rb") as fr:
        comms_test = pkl.load(fr)

    return prepare_test(srcs_test, nodes_test, edges_test, comms_test)


if __name__ == "__main__":
    sub_data_folder = "smart_contracts/comms_4_20"
    val_set = test_data_prepare(sub_data_folder)
    iterator = val_set.as_numpy_iterator()
    for next_element in iterator:
        print(next_element[0][0])
        print(next_element[1][0])
        print(next_element[2][0])
        print(next_element[3][0])
        break

"""
这是一个用于数据预处理的 Python 代码。它主要包括以下几个部分：

1. 定义了一些常量和参数，例如 BUFFER_SIZE, BATCH_SIZE, MAX_LENGTH_SBT 等等。
2. 实现了一个cut_if_longer函数，用于对给定的输入序列进行截断。
3. 实现了 prepare_train_val_test 函数，用于准备训练集、验证集和测试集数据。其中，用到了 TensorFlow 中的 Dataset.from_generator() 和 mapped Dataset 等函数。
4. 实现了 data_prepare() 和 test_data_prepare() 函数，分别用于准备训练集和测试集的数据。
5. 在主函数中调用 test_data_prepare() 函数并遍历测试集数据，输出每个样本的第一个元素（即 sbt_seq）。

这段代码的作用是将原始数据转化为模型可以输入的格式。具体来说，它将每个合约（smart contract）表示成一个一维序列（sbt_seq），以及一个节点序列（node）、边的矩阵（edge）、和对应的注释（comment）。这样处理后，模型就可以读取这些数据进行训练、验证和测试。
"""
