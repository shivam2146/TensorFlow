import tensorflow as tf
W= tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)
x= tf.placeholder(tf.float32)
y= tf.placeholder(tf.float32)
linear_model= tf.add(tf.multiply(W,x),b)
squared_delta= tf.square(tf.subtract(linear_model,y))
loss = tf.reduce_sum(squared_delta)
weights= tf.Variable(tf.random_normal([300,200],stddev=0.5),name="weights")
init = tf.global_variables_initializer()   #variables need to be initialized
weights= weights +3;
with tf.Session() as sess:
    sess.run(init)
    #print(linear_model)
    f= {x:[1,2,3,4],y:[0,-1,-2,-3]}
    o = sess.run(loss,feed_dict= f)
    print(o)
    r= sess.run(weights)
    print(r)
    writer=tf.summary.FileWriter('./my_graph',sess.graph)
    writer.close()
