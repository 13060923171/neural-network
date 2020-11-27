import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#Fetch:可以在session中同时计算多个tensor或执行多个操作
#定义三个常量
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
#加法op
add = tf.add(input2,input3)
#乘法op
mul = tf.multiply(input1,add)

with tf.compat.v1.Session()as sess:
    result,result1 = sess.run([mul,add])
    print(result,result1)


#Feed：先定义占位符，等需要的时候再传入数据
input4 = tf.compat.v1.placeholder(tf.float32)
input5 = tf.compat.v1.placeholder(tf.float32)

#乘法op
output = tf.multiply(input4,input5)

with tf.compat.v1.Session() as sess:
    print(sess.run(output,feed_dict={input4:8.0,input5:2.0}))