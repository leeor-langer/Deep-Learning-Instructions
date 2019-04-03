'''
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

curr_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_dir, 'Dataset')
mnist = input_data.read_data_sets(data_dir, one_hot=True)
batch_size = 64

batch_x, batch_y = mnist.train.next_batch(batch_size)
testX = mnist.test.images
testY = mnist.test.labels

num = np.random.randint(0, np.shape(testY)[0])
image = np.reshape(testX[num, :], (28, 28))

# Sobel Filter
sobel = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
filtered1 = np.abs(convolve2d(image, sobel, mode='same'))
filtered2 = np.abs(convolve2d(image, np.transpose(sobel), mode='same'))

# Plot
plt.subplot(3, 1, 1)
plt.imshow(image,  cmap='gray')
plt.title('Example Image, label=' + str(np.argmax(testY[num])))

plt.subplot(3, 1, 2)
plt.imshow(filtered1,  cmap='gray')
plt.title('Filtered Sobel 1' )

plt.subplot(3, 1, 3)
plt.imshow(filtered2,  cmap='gray')
plt.title('Filtered Sobel 2' )
plt.show()