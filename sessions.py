import tensorflow as tf
a= tf.constant([[1.,5.3,1.2,3.]],tf.float32,name="input_1")
b= tf.constant([[4.,12.,1.5,2.1]],tf.float32,name="input_2")
c= tf.multiply(a,b)
print(c)

with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:   #logs which cpu gets assigned
    o= sess.run(c)
    print(o)
