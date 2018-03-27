import tensorflow as tf
def try1():
    with tf.Graph().as_default():
        scalar =  tf.zeros([])
        vector = tf.zeros([5])
        matrix = tf.zeros([2,3])
        with tf.Session() as sess:
            print('shape = ',scalar.get_shape(),'and value = ',scalar.eval())
            print('shape = ',vector.get_shape(),'and value = ',vector.eval())
            print('shape = ',matrix.get_shape(),'and value = ',matrix.eval())

def broadcasting():
    with tf.Graph().as_default():
        primes = tf.constant([2,3,5,7,11,13],dtype=tf.int32)
        ones = tf.constant(1,dtype=tf.int32)
        beyond = tf.add(primes,ones)
        with tf.Session() as sess:
            print(beyond.eval())
def matmul():
    with tf.Graph().as_default():
        x= tf.constant([[5,2,4,3],[5,1,6,-2],[-1,3,-1,-2]],dtype=tf.int32) #3*4 matrix
        y= tf.constant([[2,2],[3,5],[4,5],[1,6]],dtype=tf.int32)    #4*2 matrix
        result  = tf.matmul(x,y)
        with tf.Session() as sess:
            print(result.eval())



if __name__ == '__main__':
    try1()
    broadcasting()
    matmul()
