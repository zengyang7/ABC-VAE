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

parameters = parameters/2000
# normalization
train_data = (temperature-np.min(temperature)+1)/(1.2*(np.max(temperature)-np.min(temperature)))
#train_data = temperature/np.max(temperature)


## setting
# number of basis vector of PCA and autoencoder
num = 2

# learning rate of autoencoder
learning_rate1 = 0.0001

# learning rate of NN from parameters to reduced coefficients
learning_rate2 = 0.001

# batch size
batch_size = 64

# epoch for traing autoencoder
epoch1 = 40000

# epoch for training NN from parameters to reduced coefficients
epoch2 = 50000

## PCA
mu = temperature.mean(axis=0)
U,s,V = np.linalg.svd(temperature-mu, full_matrices=False)
Zpca = np.dot(temperature - mu, V.transpose())

Rpca = np.dot(Zpca[:,:num],V[:num, :])+mu   # reconstruction
err = np.sum((temperature-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with ' + str(num)+ ' PCs:'+str(round(err, 5)));

## AE
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
    """ 
    Activation function. 
    """
    return tf.maximum(x, tf.multiply(x, 0.2))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

E_W1 = tf.Variable(xavier_init([temperature.shape[1], 512]))
E_b1 = tf.Variable(tf.zeros(shape=[512]))

E_W2 = tf.Variable(xavier_init([512, 256]))
E_b2 = tf.Variable(tf.zeros(shape=[256]))

E_W3 = tf.Variable(xavier_init([256, 128]))
E_b3 = tf.Variable(tf.zeros(shape=[128]))

E_W4 = tf.Variable(xavier_init([128, num]))
E_b4 = tf.Variable(tf.zeros(shape=[num]))

# Building the encoder
def Encoder(x):
    x = tf.matmul(x, E_W1)+E_b1
    #x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.matmul(x, E_W2)+E_b2
    #x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.matmul(x, E_W3)+E_b3
    #x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.matmul(x, E_W4)+E_b4
    x = tf.nn.tanh(tf.layers.batch_normalization(x))               
    return x
    
D_W1 = tf.Variable(xavier_init([num, 128]))  
D_b1 = tf.Variable(tf.zeros(shape=[128])) 

D_W2 = tf.Variable(xavier_init([128, 256]))  
D_b2 = tf.Variable(tf.zeros(shape=[256])) 

D_W3 = tf.Variable(xavier_init([256, 512]))  
D_b3 = tf.Variable(tf.zeros(shape=[512]))

D_W4 = tf.Variable(xavier_init([512, temperature.shape[1]]))  
D_b4 = tf.Variable(tf.zeros(shape=[temperature.shape[1]]))

theta = [E_W1, E_W2, E_W3, E_W4, E_b1, E_b2, E_b3, E_b4,
           D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]
  
def Decoder(x):
    x = tf.matmul(x, D_W1)+D_b1
    #x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.matmul(x, D_W2)+D_b2
    #x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.matmul(x, D_W3)+D_b3
    #x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.nn.sigmoid(x = tf.matmul(x, D_W4)+D_b4)
    return x

P_W1 = tf.Variable(xavier_init([parameters.shape[1], 32]))
P_b1 = tf.Variable(tf.zeros(shape=[32]))

P_W2 = tf.Variable(xavier_init([32, 32]))
P_b2 = tf.Variable(tf.zeros(shape=[32]))

P_W3 = tf.Variable(xavier_init([32, num]))
P_b3 = tf.Variable(tf.zeros(shape=[num]))

theta_P = [P_W1, P_W2, P_W3, P_b1, P_b2, P_b3]
# parameter to encoder

def Para2Enc(x):
    keep_prob = 0.6
    x = tf.matmul(x, P_W1)+P_b1
    x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.matmul(x, P_W2)+P_b2
    x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.layers.dropout(x, keep_prob)
    x = tf.matmul(x, P_W3)+P_b3
    x = tf.nn.tanh(tf.layers.batch_normalization(x))
    return x

T_input = tf.placeholder("float", [None, temperature.shape[1]])
para_input = tf.placeholder("float", [None, parameters.shape[1]])

encoder_variable = Encoder(T_input)
T_Pred = Decoder(encoder_variable)

beta = 0.9
loss_vae = tf.reduce_mean(tf.pow(T_input-T_Pred, 2))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer_vae = tf.train.AdamOptimizer(learning_rate1, beta).minimize(loss_vae, var_list=theta)

encoder_pred = Para2Enc(para_input)
T_P_Pred = Decoder(encoder_pred)
loss_para2enc = tf.reduce_sum(tf.pow(encoder_variable-encoder_pred, 2))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer_para2enc = tf.train.AdamOptimizer(learning_rate2,beta).minimize(loss_para2enc, var_list=theta_P)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1, epoch1+1):
    batch_x, _ = next_batch(batch_size, parameters, train_data)
    _, l = sess.run([optimizer_vae, loss_vae], feed_dict={T_input:batch_x})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
        
# Testing
# Encode and decode images from test set and visualize their reconstruction.
Ze_vae, Re_vae = sess.run([encoder_variable, T_Pred], feed_dict={T_input:train_data})
Pred_vae = Re_vae*(np.max(temperature)-np.min(temperature))*1.2+np.min(temperature)-1
err_vae = np.sum((temperature-Pred_vae)**2)/Re_vae.shape[0]/Re_vae.shape[1]
print('VAE reconstruction error with 2 PCs:'+str(round(err_vae, 6)));
delta = Pred_vae-temperature
delta_percent = delta/temperature
print(np.max(np.abs(delta_percent)))

for i in range(1, epoch2+1):
    batch_x, batch_y = next_batch(batch_size, parameters, train_data)
    _, l1 = sess.run([optimizer_para2enc, loss_para2enc], 
                     feed_dict={T_input:batch_x, para_input:batch_y})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l1))

Ze_vae_s, Re_vae_s = sess.run([encoder_pred, T_P_Pred], feed_dict={T_input:train_data, para_input:parameters})
Pred_vae_s = Re_vae_s*(np.max(temperature)-np.min(temperature))*1.2+np.min(temperature)-1
err_vae_s = np.sum((temperature-Pred_vae_s)**2)/Re_vae_s.shape[0]/Re_vae_s.shape[1]
print('VAE reconstruction error with 2 PCs:'+str(round(err_vae_s, 6)));
delta = Pred_vae_s-temperature
delta_percent = delta/temperature
print(np.max(np.abs(delta_percent)))