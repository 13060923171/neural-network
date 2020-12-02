from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#批次大小
batch_size = 64

#计算一个周期一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.compat.v1.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.compat.v1.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        #创建一个简单的神经网络：784-10
        w = tf.Variable(tf.compat.v1.truncated_normal([784,10],stddev=0.1))
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10])+0.1)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,w)+b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    #二次代价函数
    loss = tf.losses.mean_squared_error(y,prediction)
with tf.name_scope('train'):
    #使用梯度下降法
    train = tf.compat.v1.train.GradientDescentOptimizer(0.3).minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct'):
        #结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.compat.v1.Session() as sess:
    #变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter('logs/',sess.graph)
    for epoch in range(21):
        for batch in range(n_batch):
            #获取一个批次标签和数据
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        #每训练一个周期做一次测试
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter'+str(epoch)+',Testing Accuracy'+str(acc))