import tensorflow as tf
from Encoder.Built_Transformer import Built_Transformer as utils
from Encoder.EncoderModel import EncoderModel


class CodeBERT_Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(CodeBERT_Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderModel(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mha_mask):

        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # only encoding to the seq_len
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
          x, mha_attn = self.enc_layers[i](x, training, mha_mask)
        return x, mha_attn  # (batch_size, input_seq_len, d_model)
"""
这段代码定义了一个 EncoderSBT 类，用于实现 Transformer 模型中的编码器部分。

1. 在 `__init__` 函数中，定义了该类的各个参数和层。其中包括输入词嵌入层（`tf.keras.layers.Embedding`），位置编码（positional encoding），多个编码器层（`EncoderLayer`），以及一个 Dropout 层。

2. 在 `call` 函数中，首先获取输入张量 x 的序列长度 seq_len。

3. 然后，通过词嵌入层将输入 x 进行嵌入，得到嵌入张量 x。嵌入张量与 sqrt(d_model) 相乘后再加上位置编码（只取前 seq_len 个位置编码），然后应用 Dropout 层进行随机失活处理。

4. 接下来，通过循环调用编码器层（EncoderLayer）进行特征提取和表示学习。每个编码器层的输入是上一层的输出 x。在循环过程中，还返回了最后一层编码器层的注意力权重（mha_attn）。

这段代码与前面的代码有些相似，但存在一些区别：
- 这里的编码器没有使用 GCNLayer 进行图卷积操作，而是直接对输入进行词嵌入。
- 最终的输出形状为 (batch_size, input_seq_len, d_model)，与之前的代码有所不同。

请注意，在这段代码中，还依赖了其他模块，如 `TransformerUtils` 中的函数和 `EncoderLayer` 这个自定义层。你需要确保这些模块和类已经正确导入和定义。
"""
"""
在代码中，通过词嵌入层将输入 x 进行嵌入，意味着将输入 x 中的每个词转换为实数向量表示。词嵌入是一种常用的技术，用于将离散的词语映射到连续的向量空间，以便计算机能够更好地理解和处理文本数据。
词嵌入层（`tf.keras.layers.Embedding`）在代码中使用了一个嵌入矩阵，该矩阵的每一行代表一个词的嵌入向量。
输入 x 是一个包含词语索引的张量，通过词嵌入层，每个词语索引会被映射为对应的嵌入向量。这样，输入 x 中的每个词语都会被替换为其对应的嵌入向量，形成嵌入张量 x。
嵌入张量 x 的形状为 (batch_size, input_seq_len, d_model)，其中 batch_size 是输入的批次大小，
input_seq_len 是输入序列的长度，d_model 是嵌入的维度。这个嵌入张量将作为编码器的输入，在接下来的处理中进行特征提取和表示学习。
"""
# import tensorflow as tf
# from transformers import BertTokenizer, BertModel
# from modules.TransformerUtils import TransformerUtils as utils
# from modules.EncoderLayer import EncoderLayer

# class EncoderSBT(tf.keras.layers.Layer):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                maximum_position_encoding, rate=0.1):
#         super(EncoderSBT, self).__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.embedding = BertModel.from_pretrained("bert-base-uncased")  # 使用BertModel作为嵌入层
#         self.pos_encoding = utils.positional_encoding(maximum_position_encoding,
#                                                 self.d_model)
#
#         self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
#                            for _ in range(num_layers)]
#
#         self.dropout = tf.keras.layers.Dropout(rate)
#
#     def call(self, x, training, mha_mask):
#         seq_len = tf.shape(x)[1]
#         # 将嵌入和位置编码相加。
#         x = self.embedding(x)[0]  # 使用BertModel的第一个输出作为嵌入向量
#         x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         # only encoding to the seq_len
#         x += self.pos_encoding[:, :seq_len, :]
#         x = self.dropout(x, training=training)
#
#         for i in range(self.num_layers):
#           x, mha_attn = self.enc_layers[i](x, training, mha_mask)
#         return x, mha_attn
