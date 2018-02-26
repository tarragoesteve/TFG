from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import exponents
import numpy as np
import tensorflow as tf

##parameters
degree = 2;
variables = 3;

#calculating all the power of input up to the disired degree
input = tf.placeholder(tf.float32, [None, variables])

# calculating the sparce matrix from the exponents
exponent = exponents.uptodegree(variables, degree)
sparcematrix = [];
for i in range(variables):
    sparcematrix.append(np.zeros(((degree + 1), len(exponent)), dtype=np.dtype('float32')))
    for j in range(len(exponent)):
        a = exponent[j][i]
        sparcematrix[i][a][j] = np.float32(1.0)

def work (batch):
    # calculating all the power of input up to degree
    power = [tf.constant(np.repeat(1, variables), tf.float32)]
    for _ in range(degree):
        power.append(tf.multiply(power[len(power) - 1], batch))

    #transpose and slice
    transposedpower = tf.transpose(power)
    singlepowers = []
    for i in range(variables):
        singlepowers.append(tf.slice(transposedpower, [i, 0], [1, degree + 1]))

    #compute monomials
    result = np.repeat(1.0, len(exponent))
    for i in range(variables):
        result = result * tf.matmul(singlepowers[i], sparcematrix[i])

    w = tf.get_variable("weights", [len(exponent)], dtype=tf.float32,initializer=tf.initializers.random_normal)


    result = w * result

    #sum monomials
    return tf.reduce_sum(result)


total = tf.map_fn(work, input)

print("Graph defined")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(total, feed_dict={input: [[1.2, 2.0, 3.1], [1.1, 2.2, 3.5]]}))