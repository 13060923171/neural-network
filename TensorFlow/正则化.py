from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_size = 64

#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义三个placeholder
x = tf.compat.v1.placeholder(tf.float32,[None,784])
y = tf.compat.v1.placeholder(tf.float32,[None,10])
keep_prob = tf.compat.v1.placeholder(tf.float32)

# #784-1000-500-10
#
w1 = tf.Variable(tf.compat.v1.truncated_normal([784,1000],stddev=0.1))
b1 = tf.Variable(tf.zeros([1000])+0.1)
l1 = tf.nn.tanh(tf.matmul(x,w1)+b1)
l1_drop = tf.nn.dropout(l1,keep_prob)

w2 = tf.Variable(tf.compat.v1.truncated_normal([1000,500],stddev=0.1))
b2 = tf.Variable(tf.zeros([500])+0.1)
l2 = tf.nn.tanh(tf.matmul(l1_drop,w2)+b2)
l2_drop = tf.nn.dropout(l2,keep_prob)

w3 = tf.Variable(tf.compat.v1.truncated_normal([500,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(l2_drop,w3)+b3)

#正则化
l2_loss = tf.nn.l2_loss(w1)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2)+tf.nn.l2_loss(w3)+tf.nn.l2_loss(b3)
#交叉熵
loss = tf.losses.categorical_crossentropy(y,prediction) + 0.0005*l2_loss
#使用梯度下降法
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化变量
init = tf.compat.v1.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中的最大的值所在的位置

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for bath in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print('Iter'+str(epoch)+',Testing Accuracy'+str(test_acc)+',Train Accuracy'+str(train_acc))

