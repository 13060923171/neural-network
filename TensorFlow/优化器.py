from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_size = 64
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.compat.v1.placeholder(tf.float32,[None,784])
y = tf.compat.v1.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,w)+b)

#交叉熵
loss = tf.losses.mean_squared_error(y,prediction)
#使用梯度下降法
train_step = tf.compat.v1.train.AdadeltaOptimizer(0.01).minimize(loss)

#初始化变量
init = tf.compat.v1.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中的最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter'+str(epoch)+'.Testing Accurary'+str(acc))