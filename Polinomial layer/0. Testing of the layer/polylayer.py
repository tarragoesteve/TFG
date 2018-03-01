import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
import exponents


class PolyLayer(base.Layer):
    def __init__(self, variables, degree=3):
        self._variables =  variables
        self._degree = degree

    def build(self, _):
        pass

    def call(self, input, **kwargs):
        #aux = tf.cast(inputs, tf.float32)
        #input = tf.reshape(aux, [None, self._variables])
        # calculating the sparce matrix from the exponents
        self._exponent = exponents.uptodegree(self._variables, self._degree)
        self._sparcematrix = [];
        for i in range(self._variables):
            self._sparcematrix.append(np.zeros(((self._degree + 1), len(self._exponent)), dtype=np.dtype('float32')))
            for j in range(len(self._exponent)):
                a = self._exponent[j][i]
                self._sparcematrix[i][a][j] = np.float32(1.0)

        def work(batch):
            # calculating all the power of input up to degree
            power = [tf.constant(np.repeat(1, self._variables), tf.float32)]
            for _ in range(self._degree):
                power.append(tf.multiply(power[len(power) - 1], batch))

            # transpose and slice
            transposedpower = tf.transpose(power)
            singlepowers = []
            for i in range(self._variables):
                singlepowers.append(tf.slice(transposedpower, [i, 0], [1, self._degree + 1]))

            # compute monomials
            result = np.repeat(1.0, len(self._exponent))
            for i in range(self._variables):
                result = result * tf.matmul(singlepowers[i], self._sparcematrix[i])

            with tf.variable_scope("foo"):
                w = tf.get_variable("weights", [len(self._exponent)], dtype=tf.float32, initializer=tf.initializers.random_normal)
                result = w * result
                # sum monomials
                return tf.reduce_sum(result)

        output = tf.map_fn(work, input)

        return output

    def layer(self, input, variables, degree=3):
        self.__init__(variables=variables,degree=degree)
        return self.call(input)