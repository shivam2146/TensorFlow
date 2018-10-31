import tensorflow as tf

def try1():
    x= tf.constant(5.2,shape=(2,2)) #Good practice to define shape as well
    y= tf.Variable([0])     #must always be assigned a default value
    y = y.assign([5])
    with tf.Session() as sess:
        initialization = tf.global_variables_initializer()
        print(y.eval())
        print(x.eval())

def try2():
    g= tf.Graph()   #tensorflow provides a default graph but we can create our own
    with g.as_default():    #establish graph as default graph
        x= tf.constant(8,name='x_const')
        y= tf.constant(5,name='y_const')
        z= tf.constant(6,name='z_const')
        my_sum= tf.add(x,y,name='x_y_sum')
        my_sum1 = tf.add(z,my_sum,name='x_y_z_sum')
        with tf.Session() as sess:
            print(my_sum1.eval())
def try3():
    with tf.Graph().as_default():
        primes= tf.constant([2,3,5,7,11,13],dtype=tf.int32)
        ones = tf.ones([6],dtype=tf.int32)
        beyond_primes = tf.add(primes,ones)
        with tf.Session() as sess:
            print(beyond_primes.eval())
if __name__ == '__main__':
    try1()
    try2()
    try3()
