import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

colors = ["r", "g"]
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
num_classes = 2

# Placeholders
X = tf.placeholder(tf.float32, [None, num_inputs], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

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
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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

        for step in range(0, train_len):

            batch_X = train_X[step * batch_size:(step + 1) * batch_size, :]
            batch_Y = train_Y[step * batch_size:(step + 1) * batch_size]

            sess.run(optimizer, feed_dict={X: batch_X, Y: batch_Y})

            # Display logs per epoch step
            if (step % display_step) == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                #print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c))

    test_X = np.loadtxt('Dataset/ticeval2000.txt')
    test_Y = np.expand_dims(np.loadtxt('Dataset/tictgts2000.txt', dtype=int), axis=1)

    res_Y = sess.run(pred, feed_dict={X: test_X})
    res_Y = np.array(np.squeeze(np.round(res_Y)), dtype=int)

    acc = np.sum(np.squeeze(test_Y, axis=1) == res_Y)/test_Y.shape[0]
    print('Final Accuracy = ' + str(acc))
    print('Final Precision = ' + str(precision_score(test_Y, res_Y)))

    # dictionary with 2d mapping per class
    hidden_layer_res = sess.run(hidden, feed_dict={X:test_X})
    res_per_class = {}
    pca = PCA(n_components=2)
    hidden_layer_res_2d = pca.fit_transform(hidden_layer_res)
    ind_0 = np.where(test_Y == 0)[0]
    ind_1 = np.where(test_Y == 1)[0]

    # Plot each class in separate color
    fig, ax = plt.subplots()
    ax.scatter(hidden_layer_res_2d[ind_0, 0], hidden_layer_res_2d[ind_0, 1], c=colors[0], label=colors[0], s=30)
    ax.scatter(hidden_layer_res_2d[ind_1, 0], hidden_layer_res_2d[ind_1, 1], c=colors[1], label=colors[1], s=100, marker = "^")

    # plt.legend()
    plt.legend()
    plt.title('Feature Space')
    plt.show()