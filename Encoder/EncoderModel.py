import tensorflow as tf
from Encoder.Multi_Head_Attention import Multi_Head_Attention
from Encoder.Built_Transformer import Built_Transformer as utils
class EncoderModel(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()

        self.mha = Multi_Head_Attention(d_model, num_heads)
        self.ffn = utils.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    def call(self, mha_x, training, mask):

        mha_output, mha_attn = self.mha(mha_x, mha_x, mha_x, mask)  # (batch_size, input_seq_len, d_model)
        mha_output = self.dropout1(mha_output, training=training)
        mha_output = self.layernorm1(mha_x + mha_output)  # (batch_size, input_seq_len, d_model)
        ffn_mha_output = self.ffn(mha_output)  # (batch_size, input_seq_len, d_model)
        ffn_mha_output = self.dropout2(ffn_mha_output, training=training)
        out_mha = self.layernorm2(mha_output + ffn_mha_output)  # (batch_size, input_seq_len, d_model)

        return out_mha, mha_attn

"""
这段代码定义了一个编码器层（EncoderLayer），用于构建Transformer模型的编码器部分。

在初始化方法中，指定了编码器层的一些参数：d_model表示模型的维度大小，num_heads表示多头注意力机制的头数，dff表示前馈网络的隐藏层大小，rate表示dropout层的丢弃率。

编码器层包含以下组件：
- self.mha：多头注意力机制（MultiHeadAttention），使用输入mha_x对自身进行注意力计算。返回输出mha_output和注意力权重mha_attn。
- self.ffn：点式前馈网络（point_wise_feed_forward_network），以mha_output为输入，通过两个全连接层进行非线性变换。返回输出ffn_mha_output。
- self.layernorm1和self.layernorm2：归一化层（LayerNormalization），用于归一化输入和输出。
- self.dropout1、self.dropout2和self.dropout3：dropout层，用于在训练过程中随机丢弃神经元以防止过拟合。

在call方法中，传入输入mha_x、训练标志training和掩蔽张量mask，进行以下操作：
- 使用self.mha对mha_x进行多头注意力计算，得到输出mha_output和注意力权重mha_attn。
- 对mha_output应用dropout层，并与输入mha_x相加后进行归一化。
- 将归一化后的结果输入到self.ffn进行前馈网络计算，得到输出ffn_mha_output。
- 对ffn_mha_output应用dropout层，并与mha_output相加后进行归一化，得到编码器层的最终输出out_mha。

该编码器层用于将输入序列进行特征提取和表示学习，为后续的解码器提供编码信息。
"""
