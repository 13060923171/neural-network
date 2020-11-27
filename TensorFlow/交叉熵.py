from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#批次大小
batch_size = 64
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络:784-10
W = tf.Variable(tf.compat.v1.truncated_normal([784,10],stddev=0.1))
b = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.losses.mean_squared_error(y,prediction)
#交叉熵
loss = tf.losses.softmax_cross_entropy(y,prediction)
#使用梯度下降法
train = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss)

#结果存放在一个布尔值列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.compat.v1.Session() as sess:
    #变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    #周期epoch：所有数据训练一次，就是一个周期
    for epoch in range(21):
        for batch in range(n_batch):
            #获取一个批次的数据和标签
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        #每训练一个周期做一个测试
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter'+str(epoch)+',Testing Accuracy'+str(acc))

import pylab
im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()