
import tensorflow as tf
from Encoder.Multi_Head_Attention import Multi_Head_Attention
from Encoder.Built_Transformer import Built_Transformer as utils

"""
DecoderLayer 继承自 tf.keras.layers.Layer，并接受一些参数：d_model 表示模型的维度，num_heads 表示多头注意力机制的头数
dff 表示前馈神经网络的隐藏层维度，rate 表示 Dropout 层的丢弃率。
"""
class DecoderModel(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderModel, self).__init__()
        self.mha1 = Multi_Head_Attention(d_model, num_heads)
        self.mha2 = Multi_Head_Attention(d_model, num_heads)

        self.ffn = utils.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, mha_x, enc_output, training, look_ahead_mask, enc_padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # self-attention in the decoder input sentence
        attn1, attn_weights_block1 = self.mha1(mha_x, mha_x, mha_x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1_mha = self.layernorm1(attn1 + mha_x)

        # attention to encoder output are applied on the out1
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1_mha, enc_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1_mha)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

"""
在 __init__ 方法中，初始化了多头注意力模块 self.mha1 和 self.mha2，使用 MultiHeadAttention 类来创建。
还创建了前馈神经网络模块 self.ffn，使用 utils.point_wise_feed_forward_network 方法来创建。同时，为了进行残差连接和层归一化，
创建了三个层归一化层 self.layernorm1、self.layernorm2、self.layernorm3，并创建了三个 Dropout 层 self.dropout1、self.dropout2、self.dropout3。
call 方法实现了解码器层的前向传播过程。它接收多个输入参数：mha_x 表示解码器的输入；enc_output 表示编码器的输出；training 表示是否处于训练模式；
look_ahead_mask 表示用于掩盖未来信息的掩码；enc_padding_mask 表示用于掩盖编码器输出中的填充部分。
在 call 方法中，首先通过 self.mha1 执行解码器自注意力机制，得到注意力权重 attn_weights_block1 和自注意力输出 attn1。
然后将 attn1 与输入 mha_x 进行残差连接，并通过 self.dropout1 和 self.layernorm1 进行层归一化，得到 out1_mha。

接下来，通过 self.mha2 执行解码器与编码器的注意力机制，得到注意力权重 attn_weights_block2 和注意力输出 attn2。将 attn2 与 out1_mha 进行残差连接，
并通过 self.dropout2 和 self.layernorm2 进行层归一化，得到 out2。

随后，通过前馈神经网络 self.ffn 处理 out2，得到 ffn_output。对 ffn_output 进行 Dropout 和层归一化，得到最终的输出 out3。

最后，将 out3、attn_weights_block1 和 attn_weights_block2 返回。

这段代码展示了 Transformer 解码器中一个解码器层的实现，包含了多头注意力机制、前馈神经网络、残差连接和层归一化等关键组件，用于实现序列到序列的任务。
"""
