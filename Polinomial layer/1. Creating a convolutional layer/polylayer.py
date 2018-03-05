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
        self._stride_rows = 1
        self._stride_cols = 1
        self._activation = activation
        self._final_width = final_width
        self._final_height = final_height
        self._input_width = input_width
        self._input_height = input_height
        self._exponent = exponents.uptodegree(self._variables, self._degree)
        self._sparcematrix = []
        self._weights = tf.get_variable("my_weights", [self._filters, len(self._exponent)],dtype=tf.float32, initializer=tf.random_normal_initializer)
        for i in range(self._variables):
            #tf.SparseTensor(indices=,values=np.repeat(1,),dense_shape=[len(self._exponent)])
            self._sparcematrix.append(np.zeros(((self._degree + 1), len(self._exponent)), dtype=np.dtype('float32')))
            for j in range(len(self._exponent)):
                a = self._exponent[j][i]
                self._sparcematrix[i][a][j] = np.float32(1.0)
        print("Number of monomials:" + str(len(self._exponent)))

    def build(self, _):
        pass

    def _compute_filter(self, variables):
        # calculating all the power of input up to degree
        power = [tf.constant(np.repeat(1, self._variables), tf.float32)]
        for _ in range(self._degree):
            power.append(tf.multiply(power[len(power) - 1], variables))

        transposedpower = tf.transpose(power)
        singlepowers = []
        for i in range(self._variables):
            singlepowers.append(tf.slice(transposedpower, [i, 0], [1, self._degree + 1]))

        # compute monomials #TODO: apply scan to this part
        result = np.repeat(1.0, len(self._exponent))
        for i in range(self._variables):
            result = result * tf.matmul(singlepowers[i], self._sparcematrix[i])

        #multiply monomials per weights and apply activation function
        return self._activation(tf.reduce_sum(result*self._weights, 1))

    def call(self, input, **kwargs):
        #Input = [batch, height, width, chanels]
        #Patches = [batch, height, width, self._kernel_size[1],self._kernel_size[0]*chanels]
        patches = tf.extract_image_patches(images=input,ksizes=[1,self._kernel_size[1],self._kernel_size[0], self._channels], strides=[1, self._stride_rows, self._stride_cols, 1], rates=[1,1,1,1], padding=self._padding)
        ksize = self._kernel_size[1] * self._kernel_size[0] * self._channels
        reshaped = tf.reshape(tensor=patches, shape=[-1, ksize])
        return tf.reshape(tensor=tf.map_fn(fn=self._compute_filter,elems=reshaped,parallel_iterations=10000), shape=[-1, self._final_height, self._final_width, self._filters])



  #Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel

# X = tf.placeholder(dtype=tf.float32)
#
# input_layer = tf.reshape(X, [-1, 28, 28, 1])
#
# myconv = Conv2DPolynomial(name="conv1",filters=32, channels =1,
#                             kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu, degree=1,
#                             final_width=28, final_height=28, input_width=28, input_height=28)# output = mylayer.call(input)
#
# print("layer created")
# output = myconv.call(input_layer)
# with tf.Session() as sess:
# # Run the initializer
#     sess.run(tf.global_variables_initializer())
#     mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#     train_data = mnist.train.images  # Returns np.array
#     print(sess.run(output,feed_dict={X:  train_data[0:50]}))#[[[[3.1], [4.1], [5.1]], [[1.1], [2.1], [3.3]]]]}))#[[[3.1, 1.2], [4.1, 1.2], [5.1,1.2]], [[1.1,1.2], [2.1,2.2], [3.3,1.2]]]}))
