import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
import exponents


class PolyLayer(base.Layer):
    def __init__(self, variables, degree=3):
        self.variables =  variables
        self.degree = degree

    def build(self, _):
        pass

    def call(self, inputs, **kwargs):
        input = tf.placeholder(tf.float32, [None, self.variables])

        # calculating the sparce matrix from the exponents
        exponent = exponents.uptodegree(self.variables, self.degree)
        sparcematrix = [];
        for i in range(self.variables):
            sparcematrix.append(np.zeros(((self.degree + 1), len(exponent)), dtype=np.dtype('float32')))
            for j in range(len(exponent)):
                a = exponent[j][i]
                sparcematrix[i][a][j] = np.float32(1.0)

        def work(batch):
            # calculating all the power of input up to degree
            power = [tf.constant(np.repeat(1, self.variables), tf.float32)]
            for _ in range(self.degree):
                power.append(tf.multiply(power[len(power) - 1], batch))

            # transpose and slice
            transposedpower = tf.transpose(power)
            singlepowers = []
            for i in range(self.variables):
                singlepowers.append(tf.slice(transposedpower, [i, 0], [1, degree + 1]))

            # compute monomials
            result = np.repeat(1.0, len(exponent))
            for i in range(self.variables):
                result = result * tf.matmul(singlepowers[i], sparcematrix[i])

            w = tf.get_variable("weights", [len(exponent)], dtype=tf.float32, initializer=tf.initializers.random_normal)
            result = w * result

            # sum monomials
            return tf.reduce_sum(result)

        output = tf.map_fn(work, input)

        return output