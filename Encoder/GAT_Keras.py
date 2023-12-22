import tensorflow as tf
from tensorflow.keras import activations

class GAT_Keras(tf.keras.layers.Layer):
    """
            初始化 GCNLayer 层的构造函数。

            参数：
            - units: 输出特征的维度。
            - activation: 激活函数的名称，默认为 'relu'。
            - initializer: 权重矩阵的初始化方法，默认为 'glorot_uniform'。
            - sparse: 是否使用稀疏张量（SparseTensor），默认为 False。
            - use_bias: 是否使用偏置项，默认为 True。
            - **kwargs: 其他可选的 Layer 参数。
            """
    def __init__(self, units, activation='relu', initializer='glorot_uniform', sparse=False, use_bias=True, **kwargs):
        self.activation = activations.get(activation)
        self.output_dim = units
        self.initializer = initializer
        self.sparse = sparse
        self.use_bias = use_bias

        super(GAT_Keras, self).__init__(**kwargs)

    def build(self, input_shape):
        """
                构建 GCNLayer 层的权重。

                参数：
                - input_shape: 输入数据的形状，一个包含两个元素的列表，分别是节点特征和边信息的形状。
                """
        self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[0][-1], self.output_dim),
                                          initializer=self.initializer,
                                          trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                              shape=(self.output_dim,),
                                              initializer='zeros',
                                              trainable=True)
        else:
            self.bias = None

        super(GAT_Keras, self).build(input_shape) #在Keras模型中构建一个神经网络并输出特征

    def call(self, x):
        """
               定义 GCNLayer 的前向传播操作。

               参数：
               - x: 一个包含两个张量的列表，分别是节点特征和邻接矩阵（边信息）。

               返回：
               - output: GCNLayer 的输出张量。
               """
        assert isinstance(x, list)
        # # Get shapes of our inputs and weights
        nodes, edges = x
        nodes = tf.cast(nodes, tf.float32)
        edges = tf.cast(edges, tf.float32)
        edges += tf.eye(tf.shape(edges)[1])
        output = tf.matmul(edges, nodes)
        output = tf.matmul(output, self.kernel)

        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        """
                计算 GCNLayer 的输出形状。

                参数：
                - input_shape: 输入数据的形状。

                返回：
                - 输出形状的元组。
                """
        assert isinstance(input_shape, list)
        return (None,input_shape[0][1], self.output_dim)

    def get_config(self):
        """
                获取层的配置信息，以便序列化模型。

                返回：
                - 包含配置信息的字典。
                """
        config = {
            'units': self.output_dim,
            'activation': activations.serialize(self.activation),
        }

        base_config = super(GAT_Keras, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""
这个 GCNLayer 类定义了一个 GCN 层，用于处理图数据。它包括初始化方法、权重构建方法、前向传播方法以及用于获取配置信息的方法。
这个层可以被添加到 Keras 模型中，以构建图卷积网络。它接受节点特征和邻接矩阵作为输入，并产生 GCN 的输出特征。"""