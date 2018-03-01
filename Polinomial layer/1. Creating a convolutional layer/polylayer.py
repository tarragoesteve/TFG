import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
import exponents


class Conv2DPolynomial(base.Layer):
    def __init__(self, degree=1,filters=3,kernel_size=[3, 3],channels =1,
                 padding="same",activation=tf.nn.relu):
        self._variables = kernel_size[1] * kernel_size[0] * channels
        self._channels = channels
        self._degree = degree
        self._kernel_size = kernel_size
        self._filters = filters
        self._padding = padding
        self._activation = activation
        self._final_width = 3
        self._final_height = 2
        self._input_width = 3
        self._input_height = 3
        self._exponent = exponents.uptodegree(self._variables, self._degree)
        self._sparcematrix = [];
        self._weights = [];
        for i in range(self._filters):
            self._weights.append(tf.get_variable("w"+str(i), [len(self._exponent)], dtype=tf.float32, initializer=tf.random_normal_initializer))
        for i in range(self._variables):
            self._sparcematrix.append(np.zeros(((self._degree + 1), len(self._exponent)), dtype=np.dtype('float32')))
            for j in range(len(self._exponent)):
                a = self._exponent[j][i]
                self._sparcematrix[i][a][j] = np.float32(1.0)

    def build(self, _):
        pass

    def _inside_input(self, x, y):
        if x < 0 : return False
        if y < 0 : return False
        if x >= self._input_width: return False
        if y >= self._input_height: return False
        return True

    def _compute_filter(self, input,x,y):
        variables = []
        for i in range(x-self._kernel_size[1]/2,x+1+self._kernel_size[1]/2):
            for j in range(y-self._kernel_size[0]/2,y+1+self._kernel_size[0]/2):
                if self._inside_input(x,y):
                    variables.append(input[x][y][:])
                else:
                    variables.append(np.repeat(0,self._channels))
        flatvar = tf.reshape(variables,[-1])
        # calculating all the power of input up to degree
        power = [tf.constant(np.repeat(1, self._variables), tf.float32)]
        for _ in range(self._degree):
            power.append(tf.multiply(power[len(power) - 1], flatvar))

        # transpose and slice
        transposedpower = tf.transpose(power)
        singlepowers = []
        for i in range(self._variables):
            singlepowers.append(tf.slice(transposedpower, [i, 0], [1, self._degree + 1]))

        # compute monomials
        result = np.repeat(1.0, len(self._exponent))
        for i in range(self._variables):
            result = result * tf.matmul(singlepowers[i], self._sparcematrix[i])
        ret = []
        for w_variable in self._weights:
            ret.append(tf.reduce_sum(result*w_variable))
        return ret


    def call(self, input, **kwargs):
        aux = []
        for i in range(self._final_height):
            for j in range(self._final_width):
                aux.append(self._compute_filter(input, i, j))

        output = tf.reshape(aux, [self._final_height,self._final_width,self._filters])
        return output




input = tf.placeholder(dtype=tf.float32)
mylayer = Conv2DPolynomial()
output = mylayer.call(input)

with tf.Session() as sess:
    # Run the initializer
    sess.run(tf.global_variables_initializer())
    print(sess.run(output,feed_dict={input: [[[3.1], [4.1], [5.1]], [[1.1], [2.1], [3.3]]]}))#[[[3.1, 1.2], [4.1, 1.2], [5.1,1.2]], [[1.1,1.2], [2.1,2.2], [3.3,1.2]]]}))
