import tensorflow as tf
def layer(input,weight_shape,bias_shape):
    weight_init= tf.random_uniform_initializer(minval=-1,maxval=1)
    bias_init = tf.constant_initializer(value=0)
    W= tf.get_variable("W",weight_shape,initializer=weight_init)
    b= tf.get_variable("b",bias_shape,initializer=bias_init)
    print("Printing names of weight parameters")
    print(W.name)
    print("Printing names of bias parameters")
    print(b.name)
    return tf.matmul(input,W)+b

def my_network(input):
    with tf.variable_scope("layer_1"):
        output_1= layer(input,[784,100],[100])

    with tf.variable_scope("layer_2"):
        output_2 = layer(output_1,[100,50],[50])

    with tf.variable_scope("layer_3"):
        output_3= layer(output_2,[50,10],[10])

    return output_3

def main():
    with tf.variable_scope("shared_var") as scope:
        i_1= tf.placeholder(tf.float32,[1000,784],name="i_1")
        my_network(i_1)
        scope.reuse_variables()             #allows sharing/resuing of variables otherwise would have thrown error
        i_2= tf.placeholder(tf.float32,[1000,784],name="i_2")
        my_network(i_2)
if __name__ == "__main__":
    main()
