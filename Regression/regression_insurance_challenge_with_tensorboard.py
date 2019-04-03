import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(curr_path, 'Logs')
rng = np.random

data = np.loadtxt('Dataset/ticdata2000.txt')
train_X = data[:, :-1]
train_Y = np.expand_dims(data[:, -1], axis=1)

# Parameters
learning_rate = 0.001
training_epochs = 1000
display_step = 600
num_inputs = 85
batch_size = 50
train_len = np.int(np.floor(len(train_Y) / batch_size))

# Placeholders
X = tf.placeholder(tf.float32, shape=[None, num_inputs], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

# Set model weights
W1 = tf.get_variable("W1", [num_inputs, 10], initializer=tf.random_normal_initializer())
b1 = tf.get_variable("b1", [10], initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2", [10, 1], initializer=tf.random_normal_initializer())
b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())
global_step = tf.Variable(0, trainable=False, name='global_step')

# Construct a linear model
hidden = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(X, W1), b1))
pred = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden, W2), b2))

# Mean squared error
cost = tf.losses.mean_squared_error(pred, Y)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create summaries
tf.summary.scalar("loss", cost)
tf.summary.histogram("Weights1", W1)
tf.summary.histogram("Biases1", b1)
tf.summary.histogram("Weights2", W2)
tf.summary.histogram("Biases2", b2)

merged_summary_op = tf.summary.merge_all()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Fit all training data
    # Begin Training
    rnd_perm = np.arange(len(train_Y))
    for epoch in range(training_epochs):

        # scramble training data
        rnd_perm = np.random.permutation(rnd_perm)
        train_X = train_X[rnd_perm, :]
        train_Y = train_Y[rnd_perm]

        for step in range(0, train_len):

            batch_X = train_X[step*batch_size:(step+1)*batch_size, :]
            batch_Y = train_Y[step*batch_size:(step+1)*batch_size]

            sess.run(optimizer, feed_dict={X: batch_X, Y: batch_Y})

            # Display logs per epoch step
            if (step % display_step) == 0:
                c, summary = sess.run([cost, merged_summary_op], feed_dict={X: train_X, Y:train_Y})

                # Write logs at every iteration
                summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

                #print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c))



    #print("W1=" + str(sess.run(W1)))