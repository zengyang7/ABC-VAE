#!/usr/bin/env python
# Copyright 2018 YangZeng
""" Training the numeric data """

# standard library imports
import os, time, sys
import scipy.io as scio

# third party imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()

## load data
#mat_file = scio.loadmat(sys.argv[1])
mat_file_path = '/Users/zengyang/VAE/demo/6_nonlinear/sensitive_data.mat'
mat_file = scio.loadmat(mat_file_path)

parameters = mat_file['parameter_space']
temperature = mat_file['T_sensitive'].T

# normalization
temperature = temperature/np.max(temperature)

## setting
# number of basis vector of PCA and VAE
num = 2

# Training Parameters
learning_rate = 0.0001

# batch size
batch_size = 128

# epoch 
epoch = 10000

# PCA
mu = temperature.mean(axis=0)
U,s,V = np.linalg.svd(temperature-mu, full_matrices=False)
Zpca = np.dot(temperature - mu, V.transpose())

Rpca = np.dot(Zpca[:,:num],V[:num, :])+mu   # reconstruction
err = np.sum((temperature-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with ' + str(num)+ 'PCs:'+str(round(err, 5)));

## VAE

# encoder

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

def lrelu(x):
    """ Activation function. """
    return tf.maximum(x, tf.multiply(x, 0.2))

# Building the encoder
def Encoder(x, isTrain=True, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        x = tf.layers.dense(x, 1024)
        #x = tf.layers.dropout(x, keep_prob)
        x = lrelu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.dense(x, 512, activation='relu')
        #x = tf.layers.dropout(x, keep_prob)
        x = lrelu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.dense(x, 256, activation='relu')
        #x = tf.layers.dropout(x, keep_prob)
        x = lrelu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.dense(x, num, activation='tanh')                     
        return x
    
def Decoder(x, isTrain=True, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse):
        x = tf.layers.dense(x, 128)
        #x = tf.layers.dropout(x, keep_prob)
        x = lrelu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.dense(x, 512)
        #x = tf.layers.dropout(x, keep_prob)
        x = lrelu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.dense(x, temperature.shape[1], activation='sigmoid')
        return x
    
# parameter to encoder
def Para2Enc(x, isTrain=True, reuse=False):
    keep_prob = 0.6
    with tf.variable_scope('para2enc', reuse=reuse):
        x = tf.layers.dense(x, 32, activation='elu')
        x = tf.layers.dropout(x, keep_prob)
        x = lrelu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.dense(x, num, activation='tanh')
        return x

T_input = tf.placeholder("float", [None, temperature.shape[1]])
para_input = tf.placeholder("float", [None, parameters.shape[1]])

isTrain_vae = tf.placeholder(dtype=tf.bool)
isTrain_para = tf.placeholder(dtype=tf.bool)

encoder_variable = Encoder(T_input, isTrain=isTrain_vae)
T_Pred = Decoder(encoder_variable, isTrain=isTrain_vae)

loss_vae = tf.reduce_mean(tf.pow(T_input-T_Pred, 2))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer_vae = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_vae)

#encoder_pred = Para2Enc(para_input, isTrain=isTrain_para)
#T_P_Pred = Decoder(encoder_pred, isTrain=isTrain_vae, reuse=tf.AUTO_REUSE)
#loss_para2enc = tf.reduce_mean(tf.pow(T_input-T_P_Pred, 2))
#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#    optimizer_para2enc = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_para2enc)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1, epoch+1):
    batch_x, _ = next_batch(batch_size, parameters, temperature)
    _, l = sess.run([optimizer_vae, loss_vae], feed_dict={T_input:batch_x, isTrain_vae:True})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l))

#for i in range(1, epoch+1):
#    batch_x, batch_y = next_batch(batch_size, parameters, temperature)
#    _, l1 = sess.run([optimizer_para2enc, loss_para2enc], 
#                     feed_dict={T_input:batch_x, para_input:batch_y, isTrain_vae:False, isTrain_para:True})
#    if i % 1000 == 0 or i==1:
#        print('Step %i: Minibatch Loss: %f' % (i, l1))

# Testing
# Encode and decode images from test set and visualize their reconstruction.
Ze_vae, Re_vae = sess.run([encoder_variable, T_Pred], feed_dict={T_input:temperature, isTrain_vae:False})
err_vae = np.sum((temperature-Re_vae)**2)/Re_vae.shape[0]/Re_vae.shape[1]
print('VAE reconstruction error with 2 PCs:'+str(round(err_vae, 6)));
delta = Re_vae-temperature
delta_percent=delta/temperature
print(np.max(np.abs(delta_percent)))
