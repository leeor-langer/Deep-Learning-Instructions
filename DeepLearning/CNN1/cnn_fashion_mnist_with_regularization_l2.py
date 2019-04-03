'''
Convolutional Neural Network
'''

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import itertools

colors = itertools.cycle(["r", "b", "g", "lightgreen", "lightskyblue", "lightpink", "lime", "gray", "purple", "orange"])
curr_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_dir, 'Dataset/fashion')
logs_path = os.path.join(curr_dir, 'Logs')
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 300
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 )
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
global_step = tf.Variable(0, trainable=False, name='global_step')

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])


    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #tf.summary.image('conv2', weights['wc2'])

    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out, fc1

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.get_variable('wc1', [5, 5, 1, 16], initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                           regularizer=tf.contrib.layers.l2_regularizer(scale=0.2)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.get_variable('wc2', [5, 5, 16, 16], initializer=tf.contrib.layers.xavier_initializer(uniform=True)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.get_variable('wd1', [7*7*16, 128], initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                           regularizer=tf.contrib.layers.l2_regularizer(scale=0.2)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.get_variable('wout', [128, num_classes], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
}

biases = {
    'bc1': tf.get_variable('bc1', [16], initializer=tf.zeros_initializer()),
    'bc2': tf.get_variable('bc2', [16], initializer=tf.zeros_initializer()),
    'bd1': tf.get_variable('bd1', [128], initializer=tf.zeros_initializer()),
    'out': tf.get_variable('bout', [num_classes], initializer=tf.zeros_initializer())
}

# Construct model
logits, hidden_layer = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, global_step=global_step)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create summaries
tf.summary.scalar("loss", loss_op)
tf.summary.scalar("accuracy", accuracy)
tf.summary.image('conv1', tf.transpose(weights['wc1'], perm=[3, 0, 1, 2]), max_outputs=16)
tf.summary.histogram("w1", weights['wd1'])
tf.summary.histogram("b1", biases['bd1'])
merged_summary_op = tf.summary.merge_all()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, summary = sess.run([loss_op, accuracy, merged_summary_op], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})

            # Write logs at every iteration
            summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))

    # Visualize fspace
    hidden_layer_res = sess.run(hidden_layer, feed_dict={X: mnist.test.images,
                                                         Y: mnist.test.labels})

    # dictionary with 2d mapping per class
    res_per_class = {}
    pca = PCA(n_components=2)
    hidden_layer_res_2d = pca.fit_transform(hidden_layer_res)
    for ii in range(num_classes):
        res_per_class[ii] = hidden_layer_res_2d[np.where(np.argmax(mnist.test.labels, axis=1) == ii)[0], :]

    # Plot each class in separate color
    fig, ax = plt.subplots()
    for ii in range(num_classes):
        color = next(colors)
        ax.scatter(res_per_class[ii][:, 0], res_per_class[ii][:, 1], c=color, label=color)

    # plt.legend()
    plt.legend()
    plt.title('Feature Space')
    plt.show()
