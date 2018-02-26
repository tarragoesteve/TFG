from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import exponents
import numpy as np
import tensorflow as tf

##parameters
degree = 10;
variables = 3;


#calculating all the power of input up to the disired degree
input = tf.placeholder(tf.float32)
power = [tf.constant(np.repeat(1, variables), tf.float32)]
for _ in range(degree):
    power.append(tf.multiply(power[len(power)-1], input))

#transpossing and slicing so singlepower[i] is a tensor of all powers of variable i
transposedpower = tf.transpose(power)
singlepowers = []
for i in range(variables):
    singlepowers.append(tf.slice(transposedpower,[i, 0],[1,degree+1]))

#calculating the sparce matrix from the exponents
exponent = exponents.uptodegree(variables,degree)
sparcematrix = [];
for i in range(variables):
    sparcematrix.append(np.zeros(((degree+1), len(exponent)),dtype=np.dtype('float32')))
    for j in range(len(exponent)):
        a = exponent[j][i]
        sparcematrix[i][a][j] = np.float32(1.0)

#computing all monomials
vectors = []
result = np.repeat(1.0, len(exponent),)
for i in range(variables):
    aux = tf.matmul(singlepowers[i],sparcematrix[i])
    result = aux * result;
    vectors.append(aux)

#w = tf.get_variable("weights")
#result = w * result


total = tf.reduce_sum(result);

print("Graph defined")
sess = tf.Session()
print(sess.run(total, feed_dict={input: [1, 2, 3]}))
