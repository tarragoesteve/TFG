'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from polylayer import PolyLayer

# Parameters
learning_rate = 0.001
training_epochs = 10000
display_step = 50
batch_size = 25
variables = 3

# Training Data
x = np.random.rand(100000, variables)
y = []
for i in x:
    y.append(pow(i[0], 1) * i[1] * pow(i[2], 1))
train_X = np.asarray(x[0:5000])
train_Y = np.asarray(y[0:5000])
eval_X = x[500:1000]
eval_Y = y[500:1000]

n_samples = train_X.shape[0]

#Define our model
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

mylayer = PolyLayer(variables, 3)

pred = mylayer.call(X)

# Mean squared error
loss = tf.losses.mean_squared_error(Y, pred)

#Delaring optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
print("Going to start session")
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for batch in range(n_samples/batch_size):
            X_batch = train_X[batch_size*batch:batch_size*(batch+1)]
            Y_batch = train_Y[batch_size*batch:batch_size*(batch+1)]
            sess.run(optimizer,feed_dict={X: X_batch, Y: Y_batch})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y:train_Y})
            with tf.variable_scope("foo", reuse=True):
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "weights= ",sess.run(tf.get_variable("weights",use_resource=True)) )
    print("Optimization Finished!")
    training_cost = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    #print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    #plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    #plt.legend()
    #plt.show()

    # Testing example, as requested (Issue #2)
    #test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    #test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    #print("Testing... (Mean square loss Comparison)")
    #testing_cost = sess.run(
    #    tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
    #    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    # print("Testing cost=", testing_cost)
    # print("Absolute mean square loss difference:", abs(
    #     training_cost - testing_cost))
    #
    # plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()
