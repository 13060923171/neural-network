import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次64张图片
batch_size = 64

#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
#给模型数据输入的入口起名为x-input
x = tf.placeholder(tf.float32,[None,784],name='x-input')
#给模型标签输入的入口起名为y-input
y = tf.placeholder(tf.float32,[None,10],name='y-input')

#创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
w = tf.Variable(tf.truncate_normal([784,10],stddev = 0.1))
b = tf.Variable(tf.zeros([10])+0.1)
#给模型输出起名为output
prediction = tf.nn.softmax(tf.matmul(x,w)+b,name='output')

#交叉熵代价函数
loss = tf.losses.softmax_cross_entropy(y,prediction)
#使用adam优化器，给优化器operation起名为train
train_step = tf.train.AdamOptimizer(0.001).minimize(loss,name='train')

#初始化变量
init = tf.global_variables_initializer()

#求准确率
#tf.argmax(y,1)中的1表示取y中的第1个维度中的最大值所在的位置
#tf.equal表示比较两个值是否相等，相等返回true,不相等返回false
#最后correct_prediction是一个布尔型的列表
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#tf.cast表示数据格式转换，把布尔型转化为float类型，True变成1.0，False变成为0.0
#tf.reduce_mean求平均值
#最后accuracy为准确率
#给准确率tensor起名为accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')

#定义saver用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    #变量初始化
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            #获取一个批次的数据和标签
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #喂到模型中做训练
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        #每个周期计算一次测试集准确率
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        #打印信息
        print('Iter'+str(epoch)+'.Testing Accuracy'+str(acc))
    #保存模型
    saver.save(sess,'models/my_model.ckpt')