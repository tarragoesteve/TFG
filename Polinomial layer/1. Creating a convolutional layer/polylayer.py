import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
import exponents


class Conv2DPolynomial(base.Layer):
    def __init__(self,name="convol",degree=1,filters=2,kernel_size=[3, 3],channels =1,
                 padding="same",activation=tf.nn.relu,final_width =5, final_height =2, input_width=3, input_height=3 ):
        self._variables = kernel_size[1] * kernel_size[0] * channels
        self._channels = channels
        self._degree = degree
        self._kernel_size = kernel_size
        self._filters = filters
        self._padding = padding
        self._activation = activation
        self._final_width = final_width
        self._final_height = final_height
        self._input_width = input_width
        self._input_height = input_height
        self._exponent = exponents.uptodegree(self._variables, self._degree)
        self._sparcematrix = []
        self._weights = []
        with tf.variable_scope("foo"):
            for i in range(self._filters):
                self._weights.append(tf.get_variable(name+"w"+str(i), [len(self._exponent)], dtype=tf.float32, initializer=tf.random_normal_initializer))

        for i in range(self._variables):
            #tf.SparseTensor(indices=,values=np.repeat(1,),dense_shape=[len(self._exponent)])
            self._sparcematrix.append(np.zeros(((self._degree + 1), len(self._exponent)), dtype=np.dtype('float32')))
            for j in range(len(self._exponent)):
                a = self._exponent[j][i]
                self._sparcematrix[i][a][j] = np.float32(1.0)
        print("Number of monomials:" + str(len(self._exponent)))

    def build(self, _):
        pass

    def _inside_input(self, x, y):
        if x < 0 : return False
        if y < 0 : return False
        if x >= self._input_width: return False
        if y >= self._input_height: return False
        return True

    def _compute_filter(self, variables):
        flatvar = tf.reshape(variables,[-1])
        # calculating all the power of input up to degree
        power = [tf.constant(np.repeat(1, self._variables), tf.float32)]
        for _ in range(self._degree):
            power.append(tf.multiply(power[len(power) - 1], flatvar))

        #result = []
        #for exp in self._exponent:
        #    res = 1
        #    for index in range(self._variables):
        #        res *= power[exp[index]][index]
        #    result.append(res)

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

    def _for_batch(self, input):
        allvariables = []
        for x in range(self._final_height):
            for y in range(self._final_width):
                variables = []
                for i in range(x - self._kernel_size[1] / 2, x + 1 + self._kernel_size[1] / 2):
                    for j in range(y - self._kernel_size[0] / 2, y + 1 + self._kernel_size[0] / 2):
                        if self._inside_input(i, j):
                            variables.append(input[i][j][:])
                        else:
                            variables.append(np.repeat(float(0.0), self._channels))
                allvariables.append(tf.reshape(variables, [-1]))

        allvariables = tf.reshape(allvariables, shape=[None, self._variables])

        mapped = tf.map_fn(self._compute_filter, allvariables, dtype=tf.float32)
        auxi = self._activation(mapped)
        output = tf.reshape(auxi, [self._final_height,self._final_width,self._filters])
        return output

    def call(self, input, **kwargs):
        return tf.map_fn(self._for_batch, input)



  #Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel

#X = tf.placeholder(dtype=tf.float32)

#input_layer = tf.reshape(X, [-1, 28, 28, 1])

#myconv = Conv2DPolynomial(name="conv1",filters=32, channels =1,
#                            kernel_size=[5, 5], padding="same", activation=tf.nn.relu, degree=1,
#                            final_width=28, final_height=28, input_width=28, input_height=28)# output = mylayer.call(input)

#print("layer created")
#output = myconv.call(input_layer)
#with tf.Session() as sess:
# Run the initializer
#    sess.run(tf.global_variables_initializer())
#    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#    train_data = mnist.train.images  # Returns np.array
#    print(sess.run(output,feed_dict={X:  train_data[0:2]}))#[[[[3.1], [4.1], [5.1]], [[1.1], [2.1], [3.3]]]]}))#[[[3.1, 1.2], [4.1, 1.2], [5.1,1.2]], [[1.1,1.2], [2.1,2.2], [3.3,1.2]]]}))
