'''
Author: Esteve Tarrago
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
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
