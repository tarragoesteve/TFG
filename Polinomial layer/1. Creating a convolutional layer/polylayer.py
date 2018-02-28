import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
import exponents


class Conv2DPolynomial(base.Layer):
    def __init__(self, degree=1,filters=32,kernel_size=[5, 5],
                 padding="same",activation=tf.nn.relu):
        self._variables = 5
        self._channels = 1
        self._degree = reduce(lambda x, y: x*y, kernel_size)
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
            for j in range(y-self._kernel_size[0]/2,x+1+self._kernel_size[0]/2):
                if self._inside_input(x,y):
                    variables.append(input[x][y][:])
                else:
                    variables.append(np.repeat(0,self._channels))
        #TODO: _compute_monomials variables.flatten()
        return input[0][0]


    def call(self, input, **kwargs):
        aux = []
        for i in range(self._final_height):
            for j in range(self._final_width):
                aux.append( self._compute_filter(input,i,j))
        output = aux





        def work(batch):
            #Pading

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

        #output = tf.map_fn(work, input)

        return output




input = tf.placeholder(dtype=tf.float32)
mylayer = Conv2DPolynomial()
output = mylayer.call(input)

with tf.Session() as sess:
    # Run the initializer
    print(sess.run(output,feed_dict={input: [[[3.1, 1.2], [4.1, 1.2], [5.1,1.2]], [[1.1,1.2], [2.1,2.2], [3.3,1.2]]]}))
