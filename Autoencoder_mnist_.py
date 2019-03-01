#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:13:17 2019

@author: zengyang
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

def next_batch(num, labels, U):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(labels))
    np.random.shuffle(idx)
    idx = idx[:num]
    
    U_shuffle = [U[i] for i in idx]
    label_shuffle = [labels[i] for i in idx]

    return np.asarray(U_shuffle), np.asarray(label_shuffle)

# Training Parameters
learning_rate = 0.001
num_steps = 20000
batch_size = 256

display_step = 1000
examples_to_show = 10

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
    #biases['encoder_b1']))
    x = tf.layers.dense(x, 1024, activation='relu')
    layer1 = tf.layers.dense(x, 512, activation='relu')
    layer2 = tf.layers.dense(layer1, 256, activation='relu')
    layer3 = tf.layers.dense(layer2, 6, activation='linear')                     
    # Encoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
    #biases['encoder_b2']))
    return layer3


# Building the decoder
def decoder(x):
    layer1 = tf.layers.dense(x, 256, activation='relu')
    layer2 = tf.layers.dense(layer1, 512, activation='relu')
    layer2 = tf.layers.dense(layer2, 1024, activation='relu')
    layer3 = tf.layers.dense(layer2, 784, activation='sigmoid')
    
    # Decoder Hidden layer with sigmoid activation #1
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
    #                               biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
    #                               biases['decoder_b2']))
    return layer3

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)

# Training
for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = next_batch(batch_size, y_train, x_train)

    # Run optimization op (backprop) and cost op (to get loss value)
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    # Display logs per step
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
        
# Testing
# Encode and decode images from test set and visualize their reconstruction.

Ze_vae, Re_vae = sess.run([encoder_op, decoder_op], feed_dict={X:x_test})
err_vae = np.sum((x_test-Re_vae)**2)/Re_vae.shape[0]/Re_vae.shape[1]
print('VAE reconstruction error with 2 PCs:'+str(round(err_vae, 5)));
plt.figure(figsize=(9,3))
toPlot = (x_test, Re_vae)
for i in range(10):
    for j in range(2):
        ax = plt.subplot(2, 10, 10*j+i+1)
        plt.imshow(toPlot[j][i,:].reshape(28,28), interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.tight_layout()