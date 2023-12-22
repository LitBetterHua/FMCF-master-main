

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import dot, Activation
#Transformer模型中的工具包函数
class Built_Transformer:

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, d_model):
        """
        return positional encoding
        :param position: sentence length
        :param d_model: word embedding dimension
        :return:
        """
        angle_rads = Built_Transformer.get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :], d_model)
        # 将 sin 应用于数组中的偶数索引（indices）；2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 将 cos 应用于数组中的奇数索引；2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        # cast a tf.float32 type for pos_encoding
        return tf.cast(pos_encoding, dtype=tf.float32)

    @staticmethod
    def create_padding_mask(seqs):
        """
        config the masks for each seq in seqs
        :param seqs: 2d seqs, each seq represents a sentence
        :return:
        """
        # judge whether the elements in seq are 0, if so, set 1, otherwise, set 0.
        seqs = tf.cast(tf.math.equal(seqs, 0), tf.float32)
        return seqs[:, tf.newaxis, tf.newaxis, :], seqs  # (batch_size, 1, 1, seq_len), (batch_size, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        """计算注意力权重。
        q, k, v 必须具有匹配的前置维度。
        k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
        虽然 mask 根据其类型（填充或前瞻）有不同的形状，
        但是 mask 必须能进行广播转换以便求和。

        参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成
              (..., seq_len_q, seq_len_k)。默认为None。

        返回值:
        输出，注意力权重
        """
        # transpose the matrix b (k), multiply for each corresponding head
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # 缩放 matmul_qk, dk: dimension of k
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add thew mask before come into the softmax, multiply -1e-9 to let the
        # masked element become -inf, so that when they are shifted into the softmax (see softmax formula),
        # their weights will become 0.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    @staticmethod
    def loss_function(real, pred):
        # reduction=None: individually dealing with predictions under the masks (<pad>).
        # from_logits=True: adopt softmax to return probabilities.
        # SparseCategoricalCrossentropy: for labels in numbers rather than one-hot encode.
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        # return a mean of loss
        return tf.reduce_mean(loss_)

    @staticmethod
    def create_masks(sbt_inp, node_inp, tar):
        # 编码器填充遮挡
        sbt_padding_mask, _ = Built_Transformer.create_padding_mask(sbt_inp)
        node_padding_mask, _ =Built_Transformer.create_padding_mask(node_inp)

        # 在解码器的第二个注意力模块使用。
        # 该填充遮挡用于遮挡编码器的输出。
        # dec_padding_mask = Built_Transformer.create_padding_mask(inp)

        # 在解码器的第一个注意力模块使用。
        # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
        look_ahead_mask = Built_Transformer.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask, _ = Built_Transformer.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return sbt_padding_mask, node_padding_mask, combined_mask

    @staticmethod
    def self_attention(x, y, mask):
        attn = dot([x, y], axes=[2,2])
        if mask is not None:
            attn += (mask * -1e9)
        attn = Activation('softmax')(attn)
        return dot([attn, x], axes=[2,1])

if __name__ == "__main__":
    transformer = Built_Transformer()
    print(transformer.positional_encoding(4, 3))

"""
这段代码是一个实现了Transformer模型中一些基本组件的工具类。让我们逐个函数来看一下。

1. `get_angles` 函数用于计算位置编码中的角度，它根据位置、维度和模型维度来计算角度的变化率。

2. `positional_encoding` 函数根据位置和词嵌入维度生成位置编码矩阵，该矩阵将被用作Transformer模型的输入。

3. `create_padding_mask` 函数用于创建填充遮挡（padding mask），它判断序列中的每个元素是否为零，并将为零的元素设置为1，非零的元素设置为0。

4. `create_look_ahead_mask` 函数用于创建前瞻遮挡（look ahead mask），它创建一个上三角阵，用于屏蔽未来标记。

5. `scaled_dot_product_attention` 函数计算注意力权重，并返回输出和注意力权重。

6. `point_wise_feed_forward_network` 函数创建了一个点对点前馈神经网络。

7. `loss_function` 函数定义了损失函数，用于计算模型的损失。

8. `create_masks` 函数用于创建填充遮挡和前瞻遮挡。

9. `self_attention` 函数计算自注意力机制。

在主程序中，创建了一个 TransformerUtils 的实例，并打印出位置编码的结果。

这段代码主要是为了在 Transformer 模型中提供一些实用的函数和工具。
"""

