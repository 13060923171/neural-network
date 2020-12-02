from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#批次大小
batch_size = 64

#计算一个周期一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        #平均值
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        #标准差
        tf.summary.scalar('stddev',stddev)
        #最大值
        tf.summary.scalar('max',tf.reduce_max(var))
        #最小值
        tf.summary.scalar('min',tf.reduce_min(var))
        #直方图
        tf.summary.histogram('histogram',var)

with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        #创建一个简单的神经网络：784-10
        w = tf.Variable(tf.zeros([784,10],name='W'))
        variable_summaries(w)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,w)+b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    #二次代价函数
    loss = tf.losses.mean_squared_error(y,prediction)
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #使用梯度下降法
    train = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct'):
        #结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('loss',accuracy)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            #获取一个批次标签和数据
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train],feed_dict={x:batch_xs,y:batch_ys})
        #每训练一个周期做一次测试
        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter'+str(epoch)+',Testing Accuracy'+str(acc))