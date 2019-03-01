#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:27:49 2019

@author: zengyang
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import pylab as plt

tf.reset_default_graph()
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

# PCA
mu = x_train.mean(axis=0)
U,s,V = np.linalg.svd(x_train-mu, full_matrices=False)
Zpca = np.dot(x_test - mu, V.transpose())

Rpca = np.dot(Zpca[:,:2],V[:2, :])+mu   # reconstruction
err = np.sum((x_test-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with 2 PCs:'+str(round(err, 3)));

# Autoencoder

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

def Encoder(x):
    x = tf.layers.dense(x, 512, activation='elu')
    x = tf.layers.dense(x, 128, activation='elu')
    x = tf.layers.dense(x, 2, activation='linear')
    return x

def Decoder(x):
    x = tf.layers.dense(x, 128, activation='elu')
    x = tf.layers.dense(x, 512, activation='elu')
    x = tf.layers.dense(x, 784, activation='sigmoid')
    return x

learning_rate = 0.01
batch_size = 256
epoch= 5000
x = tf.placeholder('float', [None, 784])

encoder_op = Encoder(x)
decoder_op = Decoder(encoder_op)

y_pred = decoder_op
y_true = x

loss = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()


sess = tf.Session()

sess.run(init)

for i in range(1, epoch+1):
    batch_x, _ = next_batch(batch_size, y_train, x_train)
    _, l = sess.run([optimizer, loss], feed_dict={x:batch_x})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l))


# Testing
# Encode and decode images from test set and visualize their reconstruction.
Ze_vae, Re_vae = sess.run([encoder_op, decoder_op], feed_dict={x:x_test})
err_vae = np.sum((x_test-Re_vae)**2)/Re_vae.shape[0]/Re_vae.shape[1]
print('VAE reconstruction error with 2 PCs:'+str(round(err_vae, 3)));

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title('PCA')
plt.scatter(Zpca[:5000,0], Zpca[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.subplot(122)
plt.title('Autoencoder')
plt.scatter(Ze_vae[:5000,0], Ze_vae[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.tight_layout()

plt.figure(figsize=(9,3))
toPlot = (x_test, Rpca, Re_vae)
for i in range(10):
    for j in range(3):
        ax = plt.subplot(3, 10, 10*j+i+1)
        plt.imshow(toPlot[j][i,:].reshape(28,28), interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.tight_layout()