# coding = utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np

# start create data
# 创建100个类型为np.float32的数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start #
# tf 自带的随机数，一维，范围是(-1,1)
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 初始化为0
biases = tf.Variable(tf.zeros([1]))
# 构造y
y = Weights * x_data + biases

# reduce_mean:求平均值，square:平方
loss = tf.reduce_mean(tf.square(y - y_data))
# 优化器，GradientDescentOptimizer梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 使loss函数最小
train = optimizer.minimize(loss)
# 初始化所有的变量
init = tf.initialize_all_variables()
# create tensorflow structure end #
# 创建一个会话
sess = tf.Session()
# 与变量建立连接
sess.run(init)

for step in range(1000):
    # 运行训练
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(Weights), sess.run(biases))
