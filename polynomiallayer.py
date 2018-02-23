from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sympy.polys.monomials import monomial_count

import numpy as np
import tensorflow as tf


##parameters
degree = 5;
variables = 25;
print( monomial_count(variables, degree))

##
input = tf.placeholder(tf.float32);
coeficients =  tf.get_variable("coeficients", [1, 2, 3]);


a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = input



sess = tf.Session()

print(sess.run(total, feed_dict={input: [1, 3, 4 , 5]}))
