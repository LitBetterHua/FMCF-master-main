import tensorflow as tf

class Learning_Rate(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(Learning_Rate, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    """
这段代码定义了一个自定义学习率调度器（CustomSchedule），继承自tf.keras.optimizers.schedules.LearningRateSchedule类。该调度器用于在训练深度学习模型中动态地调整学习率。

在初始化方法中，指定了模型的维度d_model和热身步数warmup_steps。其中，d_model被转换为float32类型。

调度器的__call__方法接收一个步数(step)作为输入，并根据该步数计算出一个学习率。具体计算过程如下：

1. 计算arg1，使用tf.math.rsqrt函数对步数进行反平方根操作。
2. 计算arg2，将步数乘以warmup_steps的倒数的-1.5次方。
3. 返回由d_model的平方根和arg1、arg2中较小值的乘积。

这种学习率调度器常用于在训练初期使用较小的学习率，随后逐渐增大学习率以加快模型的收敛速度。
    """
