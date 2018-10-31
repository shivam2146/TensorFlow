import tensorflow as tf
a= tf.constant(3.,name="input_a")
b= tf.constant(4.,name="input_b")
c= tf.multiply(a,b,name="mul_c")
d= tf.add(a,b,name="add_d")
e=tf.add(c,d,name="add_e")

'''
sess= tf.Session()
output= sess.run(e)
writer = tf.summary.FileWriter('./my_graph',sess.graph)
writer.close()
sess.close()
'''

#Alternative way to use tf session with scope
with tf.Session() as sess:
    sess.run(e)
    writer = tf.summary.FileWriter('./my_graph',sess.graph)
    writer.close()
