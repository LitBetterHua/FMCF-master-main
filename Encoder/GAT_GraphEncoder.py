
import tensorflow as tf
from Encoder.Built_Transformer import Built_Transformer as utils
from Encoder.EncoderModel import EncoderModel
from Encoder.GAT_Keras import GAT_Keras


class GAT_GraphEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, asthop,
               maximum_position_encoding, rate=0.1):
        super(GAT_GraphEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, name="graph_embed")
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding,
                                                self.d_model)
       # self.gcn_layer = GCNLayer(d_model)
        self.gat_layer = GAT_Keras(d_model)
        self.asthop = asthop
        self.enc_layers = [EncoderModel(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, node_input, edge_input, training, mha_mask):
        node_ebd = self.embedding(node_input)

        for i in range(self.asthop):
            node_ebd = self.gat_layer([node_ebd, edge_input])
        x = node_ebd

        seq_len = tf.shape(x)[1]
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
          x, mha_attn = self.enc_layers[i](x, training, mha_mask)
        return x, mha_attn # (batch_size, 2*input_seq_len, d_model)

"""
这段代码定义了一个 EncoderGraph 类，用于实现 Transformer 模型中的编码器部分。

1. 在 `__init__` 函数中，定义了该类的各个参数和层。其中包括输入词嵌入层（`tf.keras.layers.Embedding`），位置编码（positional encoding），图卷积层（`GCNLayer`），
多个编码器层（`EncoderLayer`），以及一个 Dropout 层。

2. 在 `call` 函数中，首先将输入的节点信息（node_input）通过词嵌入层进行嵌入，得到节点嵌入（node_ebd）。

3. 接下来，通过 GCNLayer 对节点嵌入进行 AST-Hop 操作（根据 asthop 参数控制循环次数），得到更新后的节点嵌入 x。

4. 接下来，计算节点嵌入的序列长度 seq_len，并对节点嵌入进行尺度调整和位置编码的加和操作。然后，应用 Dropout 层对结果进行随机失活处理。

5. 最后，通过循环调用编码器层（EncoderLayer）进行特征提取和表示学习。每个编码器层的输入是上一层的输出 x。在循环过程中，还返回了最后一层编码器层的注意力权重（mha_attn）。

整个过程完成了对输入数据的编码操作，将输入数据转换为更丰富的特征表示，并保留了注意力权重用于后续的任务或分析。最终的输出形状为 (batch_size, 2*input_seq_len, d_model)。

请注意，在这段代码中，还依赖了其他模块，如 `TransformerUtils` 中的函数和 `EncoderLayer`、`GCNLayer` 这两个自定义层。你需要确保这些模块和类已经正确导入和定义。
"""
