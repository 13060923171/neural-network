import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

# numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

plt.scatter(x_data, y_data)
plt.show()

# 定义两个placeholder
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 神经网络结构：1-30-1
w1 = tf.Variable(tf.compat.v1.random_normal([1, 30]))
b1 = tf.Variable(tf.zeros([30]))
wx_plus_b_1 = tf.matmul(x, w1) + b1
l1 = tf.nn.tanh(wx_plus_b_1)

w2 = tf.Variable(tf.compat.v1.random_normal([30, 1]))
b2 = tf.Variable(tf.zeros([1]))
wx_plus_b_2 = tf.matmul(l1, w2) + b2
prediction = tf.nn.tanh(wx_plus_b_2)

# 二次代价函数
loss = tf.losses.mean_squared_error(y, prediction)
# 使用梯度下降法最小化loss
train = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    # 变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    for _ in range(3000):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()