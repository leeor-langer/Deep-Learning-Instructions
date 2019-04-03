'''
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import itertools

colors = itertools.cycle(["r", "b", "g", "lightgreen", "lightskyblue", "lightpink", "lime", "gray", "purple", "orange"])
curr_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(curr_dir, 'Dataset')
logs_path = os.path.join(curr_dir, 'Logs')
mnist = input_data.read_data_sets(data_path, one_hot=True)

# Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
global_step = tf.Variable(0, trainable=False, name='global_step')

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name="layer_2")
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer, layer_2

# Construct model
logits, hidden_layer = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
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
tf.summary.histogram("Weights1", weights['h1'])
tf.summary.histogram("Weights2", weights['h2'])
tf.summary.histogram("Biases1", biases['b1'])
tf.summary.histogram("Biases2", biases['b2'])
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
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, summary = sess.run([loss_op, accuracy, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})

            # Write logs at every iteration
            summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
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
    for ii in range(num_classes):
        plt.scatter(res_per_class[ii][:, 0], res_per_class[ii][:, 1], c=next(colors))

    #plt.legend()
    plt.title('Feature Space')
    plt.show()
