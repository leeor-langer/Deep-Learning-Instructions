import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

data = np.loadtxt('Dataset/ticdata2000.txt')
train_X = data[:, :-1]
train_Y = np.expand_dims(data[:, -1], axis=1)

# Parameters
learning_rate = 0.0001
training_epochs = 500
display_step = 600
num_inputs = 85
batch_size = 50
train_len = np.int(np.floor(len(train_Y) / batch_size))


# Placeholders
X = tf.placeholder(tf.float32, [None, num_inputs], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

# Set model weights
W = tf.get_variable("W1", [num_inputs, 1], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
b = tf.get_variable("b1", [1], initializer=tf.zeros_initializer())

# Construct a linear model
pred = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(X, W), b))

# Mean squared error
cost = tf.losses.mean_squared_error(pred, Y)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    # Begin Training
    rnd_perm = np.arange(len(train_Y))
    for epoch in range(training_epochs):

        # scramble training data
        rnd_perm = np.random.permutation(rnd_perm)
        train_X = train_X[rnd_perm, :]
        train_Y = train_Y[rnd_perm]

        batch_X = train_X[0:batch_size, :]
        batch_Y = train_Y[0:batch_size]

        for step in range(0, train_len):
            sess.run(optimizer, feed_dict={X: batch_X, Y: batch_Y})

            # Display logs per epoch step
            if (step % display_step) == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                #print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c))


