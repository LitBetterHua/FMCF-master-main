
import tensorflow as tf
from Encoder.Built_Transformer import Built_Transformer as utils

class Multi_Head_Attention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention, attention_weights = utils.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

if __name__ =="__main__":
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    temp_mha = Multi_Head_Attention(d_model=256, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out)
    print("输出结果权重")
    print(attn)
"""
这段代码实现了 Transformer 模型中的多头注意力机制（Multi-head Attention）。让我们逐个函数和主程序来看一下。

1. `__init__` 函数初始化了 MultiHeadAttention 类的参数，并定义了所需的层，包括三个全连接层和一个输出层。

2. `split_heads` 函数将最后一个维度分拆为多个头，并对结果进行转置，以便得到形状为 `(batch_size, num_heads, seq_len, depth)`。

3. `call` 函数是该类的主要计算逻辑，它接收输入数据 `v`、`k`、`q` 和遮挡 `mask`，并对它们进行线性变换和拆分。然后，调用工具类中的 `scaled_dot_product_attention` 函数计算注意力权重和输出。最后，将输出进行重塑和线性变换，得到最终的输出结果。

在主程序中，创建了一个 MultiHeadAttention 的实例，并使用随机数据 `y` 进行测试，得到多头注意力的输出结果 `out` 和注意力权重 `attn`。

这段代码实现了 Transformer 模型中的关键部分，即多头注意力机制，用于对输入进行特征提取和表示学习。
"""