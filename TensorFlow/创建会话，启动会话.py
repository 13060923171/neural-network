import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#创建一个常量
m1 = tf.constant([[3,3]])
#创建一个常量
m2 = tf.constant([[2],[3]])
#矩阵乘法op
product = tf.matmul(m1,m2)
print(product)

#定义会话
sess = tf.compat.v1.Session()
result = sess.run(product)
print(result)
sess.close()

with tf.compat.v1.Session() as sess:
    result = sess.run(product)
    print(result)