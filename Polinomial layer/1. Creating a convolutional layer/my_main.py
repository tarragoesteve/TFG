'''
Author: Esteve Tarrago
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from polylayer import Conv2DPolynomial

# Parameters
learning_rate = 0.01
training_epochs = 100
display_step = 1
batch_size = 100

X = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.int32)
mode = tf.placeholder(tf.string)

"""Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
input_layer = tf.reshape(X, [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
myconv = Conv2DPolynomial(name="conv1",filters=32, channels=1,
                            kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu, degree=4,
                            final_width=28, final_height=28, input_width=28, input_height=28)
conv1 = myconv.call(input_layer)

  # conv1 = tf.layers.conv2d(
  #     inputs=input_layer,
  #     filters=32,
  #     kernel_size=[5, 5],
  #     padding="same",
  #     activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  #myconv2 = Conv2DPolynomial(filters=64,name="conv2",channels = 32, kernel_size=[5, 5], padding="same",
  #                           activation=tf.nn.relu, degree=2, final_width=28, final_height=28, input_width=28, input_height=28)
  #conv2 = myconv2.call(pool1)

conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
logits = tf.layers.dense(inputs=dropout, units=10)

predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
# Calculate Loss (for both TRAIN and EVAL modes)
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

#Declaring optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
print("Going to start session")
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)


    # Fit all training data
    for epoch in range(training_epochs):
        for batch in range(len(train_data)/batch_size):
            X_batch = train_data[batch_size*batch:batch_size*(batch+1)]
            Y_batch = train_labels[batch_size*batch:batch_size*(batch+1)]
            print("Epoch: "+ str(epoch) + "/"+ str(training_epochs)+ " Batch:"+ str(batch) + "/" + str(len(train_data)/batch_size))
            sess.run(optimizer, feed_dict={X: X_batch, mode: tf.estimator.ModeKeys.TRAIN, labels: Y_batch})
            if (batch % 100 == 0):
                with tf.variable_scope("conv1",reuse=True):
                    c = sess.run(loss, feed_dict={X: mnist.test.images[0:100], mode: tf.estimator.ModeKeys.EVAL, labels: np.asarray(mnist.test.labels[0:100], dtype=np.int32)})
                    print("cost=", "{:.9f}".format(c))
                    np.savetxt("E"+str(epoch)+"B"+str(batch)+ ".csv",sess.run(tf.get_variable("my_weights")), fmt='%.9e', delimiter=', ')
        with tf.variable_scope("conv1", reuse=True):
            c = sess.run(loss, feed_dict={X: mnist.test.images[0:100], mode: tf.estimator.ModeKeys.EVAL,
                                          labels: np.asarray(mnist.test.labels[0:100], dtype=np.int32)})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
            np.savetxt("Epoch" + str(epoch) + ".csv", sess.run(tf.get_variable("my_weights")), fmt='%.9e', delimiter=', ')
