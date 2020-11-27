import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#定义一个变量
x = tf.Variable([1,2])
#定义一个常量
a = tf.constant([3,3])
#减法op
sub = tf.subtract(x,a)
#加法op
add = tf.add(x,sub)

#所有变量初始化
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
