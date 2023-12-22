
import tensorflow as tf #一个抽象的框架，一个实体的执行分工明确的调包侠拼凑侠
from Encoder.Built_Transformer import Built_Transformer as utils
from Decoder.DecoderModel import DecoderModel

class Joint_Fusion_Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Joint_Fusion_Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers1 = [DecoderModel(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dec_layers2 = [DecoderModel(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        # self.dec_layers3 = [DecoderModel(d_model, num_heads, dff, rate)
        #                    for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dropout1 = tf.keras.layers.Dropout(rate)
    def call(self, comm_tar, sbt_output, graph_output, training,
            sbt_padding_mask, node_padding_mask, look_ahead_mask):
        seq_len = tf.shape(comm_tar)[1]
        attention_weights = {}

        comm_tar = self.embedding(comm_tar)  # (batch_size, target_seq_len, d_model)
        comm_tar *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        comm_tar += self.pos_encoding[:, :seq_len, :]
        comm_tar = self.dropout(comm_tar, training=training)

        comm_tar1 = tf.identity(comm_tar)
        comm_tar2 = tf.identity(comm_tar)


        # sbt & comm
        for i in range(self.num_layers):
            comm_tar1, block_comm, block_sbt_comm = self.dec_layers1[i](comm_tar1, sbt_output, training,
                                                 look_ahead_mask, sbt_padding_mask)
            attention_weights['decoder_layer{}_block_comm'.format(i+1)] = block_comm
            attention_weights['decoder_layer{}_sbt_comm'.format(i+1)] = block_sbt_comm

        # graph & comm
        for i in range(self.num_layers):
            comm_tar2, block_comm, block_graph_comm = self.dec_layers2[i](comm_tar2, graph_output, training,
                                                 look_ahead_mask, node_padding_mask)
            # attention_weights['decoder_layer{}_block_comm2'.format(i+1)] = block_comm
            attention_weights['decoder_layer{}_graph_comm'.format(i+1)] = block_graph_comm

        comm_tar3 = tf.concat([comm_tar1, comm_tar2], axis=-1)


        return comm_tar3, attention_weights

"""
Decoder 继承自 tf.keras.layers.Layer，并接受一些参数：num_layers 表示解码器层数，d_model 表示模型的维度，num_heads 表示多头注意力机制的头数，
dff 表示前馈神经网络的隐藏层维度，target_vocab_size 表示目标词汇表的大小，maximum_position_encoding 表示位置编码的最大长度，rate 表示 Dropout 层的丢弃率。
在 __init__ 方法中，首先将输入参数保存到类成员变量中。然后，创建了一个嵌入层 self.embedding，用于将目标序列映射为 d_model 维度的向量表示。
创建了位置编码 self.pos_encoding，使用了 utils.positional_encoding 方法来生成。接下来，创建了两个解码器层列表 self.dec_layers1 和 self.dec_layers2，
其中每个解码器层都使用 DecoderLayer 类来创建。最后，创建了两个 Dropout 层 self.dropout 和 self.dropout1。
call 方法实现了解码器的前向传播过程。它接收多个输入参数：comm_tar 表示通信目标序列，sbt_output 表示子树输出，graph_output 表示图输出，training 表示是否处于训练模式，
sbt_padding_mask 表示用于掩盖子树输出中的填充部分，node_padding_mask 表示用于掩盖图输出中的填充部分，look_ahead_mask 表示用于掩盖未来信息的掩码。
在 call 方法中，首先获取通信目标序列的长度 seq_len。然后创建一个空的字典 attention_weights，用于保存注意力权重。

接下来，通过嵌入层 self.embedding 将通信目标序列映射为向量表示。为了缩放输入，将其乘以 tf.math.sqrt(tf.cast(self.d_model, tf.float32))。然后，通过位置编码和掩码操作对通信目标序列进行处理。

接着，创建两个与 comm_tar 相同的变量 comm_tar1 和 comm_tar2。

然后，通过循环遍历解码器层数，分别对 comm_tar1 和 comm_tar2 执行解码器层的操作。注意力权重存储在 attention_weights 字典中。

最后，将 comm_tar1 和 comm_tar2 沿最后一个维度拼接起来，得到最终的输出 comm_tar3。将 comm_tar3 和 attention_weights 一起返回。

这段代码展示了通过堆叠多个解码器层构建解码器的过程，以及在不同层之间共享参数。同时，也展示了输入数据的嵌入处理和注意力权重的保存。
"""
