#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Training the numeric data """

# standard library imports
import os, time, sys
import scipy.io as scio
from scipy.stats import multivariate_normal as mvn

# third party imports
import numpy as np
import tensorflow as tf


tf.reset_default_graph()

# setting of training

'''
Input:
    training_ratio  --The ratio of training samples
    num             --The dimension of feature vector
    nor_par         --The normalization of the parameters
    N               --The number of particles of PMC
    alpha           --The ratio of PMC
    p_acc_min       --The stop critia
    
'''

file_para = open(sys.argv[1], 'r')
#file_name1 = '/Users/zengyang/VAE/demo/4_nonlinear/setting'
#file_para = open(file_name1, 'r')
list_para = file_para.readlines()
for line in list_para:
    line = line.strip('\n')
    line = line.split('#')[0].split()
    # ignore empty lines or comments
    if not line:
        continue
    # variable name
    var_name = line[0]
    # variable value
    var_value = float(line[1])
    #exec("%s = %d" % (var_name, var_value))
    exec("%s = %.4f" % (var_name, var_value))

# dimension of feature vector
num = int(num)
N = int(N)
## load data
mat_file = scio.loadmat(sys.argv[2])

# test code
#mat_file_path = '/Users/zengyang/VAE/demo/4_nonlinear/sensitive_data.mat'
#mat_file = scio.loadmat(mat_file_path)

parameters = mat_file['parameters']
temperature = mat_file['T_sensitive_4'].T

training_size = int(parameters.shape[0]*training_ratio)
print('The size of training samples: ', str(training_size))

# normalization
min_para = np.min(parameters)
max_para = np.max(parameters)
min_temp = np.min(temperature)
max_temp = np.max(temperature)

para = (parameters-min_para+2)/(max_para-min_para)
temp = (temperature-min_temp+1)/(1.2*(max_temp-min_temp))

# Training and testing temperature and parameters
train_temp = temp[0:training_size]
test_temp  = temp[training_size:-1]
train_para = para[0:training_size]
test_para  = para[training_size:-1]

# learning rate of autoencoder
learning_rate1 = 0.0001

# learning rate of NN from parameters to reduced coefficients
learning_rate2 = 0.001
beta = 0.9

# batch size
batch_size = 64

# epoch for traing autoencoder
epoch1 = 50000

# epoch for training NN from parameters to reduced coefficients
epoch2 = 50000

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

def R_squared(Prediction, Observed):
    '''
    R_squared of the prediction
    '''
    y_mean = Observed.mean(axis=0)
    SS_tot = np.sum((Observed-y_mean)**2)
    SS_res = np.sum((Prediction-Observed)**2)
    R_s = 1 - SS_res/SS_tot
    return R_s

############################### PCA ##########################################
mu_pca = train_temp.mean(axis=0)
U,s,V  = np.linalg.svd(train_temp-mu_pca, full_matrices=False)

# the reduced vector of POD
Zpca  = np.dot(test_temp - mu_pca, V.transpose())

Rpca  = np.dot(Zpca[:,:num],V[:num, :]) + mu_pca   # reconstruction
Pred_pca = Rpca*1.2*(max_temp-min_temp)+min_temp-1
err   = np.sum((Pred_pca-temperature[training_size:-1])**2)/Rpca.shape[0]/Rpca.shape[1]
R_square_pca = R_squared(Pred_pca, temperature[training_size:-1])
print('PCA reconstruction error with ' + str(num)+ ' PCs:'+str(round(err, 5)))
print('R square of PCA with '+ str(num)+ ' PCs:'+str(round(R_square_pca, 5)))

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
    '''
    Encoder
    '''
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
    #x = tf.nn.tanh(tf.layers.batch_normalization(x))               
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
    '''
    Decoder
    '''
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

P_W1 = tf.Variable(xavier_init([parameters.shape[1], 64]))
P_b1 = tf.Variable(tf.zeros(shape=[64]))

P_W2 = tf.Variable(xavier_init([64, 64]))
P_b2 = tf.Variable(tf.zeros(shape=[64]))

P_W3 = tf.Variable(xavier_init([64, num]))
P_b3 = tf.Variable(tf.zeros(shape=[num]))

theta_P = [P_W1, P_W2, P_W3, P_b1, P_b2, P_b3]
# parameter to encoder

def Para2Enc(x):
    '''
    From parameter to the reduced coefficients
    '''
    keep_prob = 0.6
    x = tf.matmul(x, P_W1)+P_b1
    x = tf.layers.dropout(x, keep_prob)
    x = tf.nn.tanh(tf.layers.batch_normalization(x))
    x = tf.matmul(x, P_W2)+P_b2
    x = tf.layers.dropout(x, keep_prob)
    x = lrelu(tf.layers.batch_normalization(x))
    x = tf.layers.dropout(x, keep_prob)
    x = tf.matmul(x, P_W3)+P_b3
    #x = lrelu(tf.layers.batch_normalization(x))
    return x

################################# Autoencoder #################################

# placeholder for Temperature field
Temp_input = tf.placeholder("float", [None, temperature.shape[1]])
# placeholder for boundary condition
para_input = tf.placeholder("float", [None, parameters.shape[1]])

# feature vector
feat_vect = Encoder(Temp_input)
# reconstruction of temperature
Temp_Pred = Decoder(feat_vect)

# loss function of autoencoder
loss_vae = tf.reduce_mean(tf.pow(Temp_input-Temp_Pred, 2))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer_vae = tf.train.AdamOptimizer(learning_rate1, beta).minimize(loss_vae, var_list=theta)

# relationship between boundary parameters and reduced coefficient
feat_pred = Para2Enc(para_input)
# temperature prediction of parameter
Temp_para_Pred = Decoder(feat_pred)
loss_para2enc = tf.reduce_sum(tf.pow(feat_vect-feat_pred, 2))

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer_para2enc = tf.train.AdamOptimizer(learning_rate2,beta).minimize(loss_para2enc, var_list=theta_P)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session()

sess.run(init)

for i in range(1, epoch1+1):
    batch_x, _ = next_batch(batch_size, train_para, train_temp)
    _, l = sess.run([optimizer_vae, loss_vae], feed_dict={Temp_input:batch_x})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
        
# Testing
# The prediction of autoencoder
Feat_vae, Re_vae = sess.run([feat_vect, Temp_Pred], feed_dict={Temp_input:test_temp})

# inverse of normalization
Temp_vae = Re_vae*(max_temp-min_temp)*1.2+min_temp-1

# average of error
err_vae = np.sum((temperature[training_size:-1]-Temp_vae)**2)/Re_vae.shape[0]/Re_vae.shape[1]
print('VAE reconstruction error with ' + str(num)+ ' PCs:'+str(round(err_vae, 5)))
delta = Temp_vae-temperature[training_size:-1]
delta_percent = delta/temperature[training_size:-1]
print(np.max(np.abs(delta_percent)))

# R square
R_square_ae = R_squared(Temp_vae, temperature[training_size:-1])
print('R square of ae with ' + str(num)+ ' PCs:'+str(round(R_square_ae, 5)))

for i in range(1, epoch2+1):
    batch_x, batch_y = next_batch(batch_size, train_para, train_temp)
    _, l1 = sess.run([optimizer_para2enc, loss_para2enc], 
                     feed_dict={Temp_input:batch_x, para_input:batch_y})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l1))

# Prediction of NN
Feat_pred,  Re_pred= sess.run([feat_pred, Temp_para_Pred], feed_dict={para_input:test_para})

# inverse of normalization
Temp_pred = Re_pred*(max_temp-min_temp)*1.2+min_temp-1

# average of error
err_vae_s = np.sum((temperature[training_size:-1]-Temp_pred)**2)/Re_pred.shape[0]/Re_pred.shape[1]
print('VAE reconstruction error with ' + str(num)+ ' PCs:'+str(round(err_vae_s, 5)))
delta = Temp_pred-temperature[training_size:-1]
delta_percent = delta/temperature[training_size:-1]
print(np.max(np.abs(delta_percent)))

# R square
R_s_ae_s = R_squared(Temp_pred, temperature[training_size:-1])
print('R square of ae with ' + str(num)+ ' PCs:'+str(round(R_s_ae_s, 5)))


############################ ABC NS ##########################################

Observations_file = scio.loadmat(sys.argv[3])
#observationname = '/Users/zengyang/VAE/demo/4_nonlinear/observation_dynamic.mat'
#Observations_file = scio.loadmat(observationname)
Observations = Observations_file['T0'].T + noise*np.random.randn(1, temp.shape[1])
obser = (Observations-min_temp+1)/(1.2*(max_temp-min_temp))

feat_obser = sess.run(feat_vect, feed_dict={Temp_input:obser})

# prior
mu_prior = np.array([50, 160, 30, 180])
num_var  = mu_prior.shape[0]
sigma_prior = np.diag((mu_prior*0.05)**2)

Kesi_record  = []
Sigma_record = []
Appro_poster = []
P_acc_record = []

data_calculation = np.zeros([N, parameters.shape[1]+2])
data_resampling  = np.zeros([N, parameters.shape[1]+1])

for i in range(N):
    theta       = np.random.multivariate_normal(mu_prior, sigma_prior, 1)
    theta_input = (theta-min_para+2)/(max_para-min_para)
    feat_theta  = sess.run(feat_pred, feed_dict={para_input:theta_input})
    pho         = np.sum((feat_theta - feat_obser)**2)
    data_calculation[i, 0:num_var] = theta
    data_calculation[i, num_var] = pho

# sort the samples increasingly with pho  
index = np.argsort(data_calculation[:, num_var])
data_calculation = data_calculation[index]

kesi  = data_calculation[int(N*alpha), num_var]

for i in range(int(N*alpha)):
    pho = data_calculation[i, num_var]
    w   = 1-(pho/kesi)**2
    data_calculation[i, num_var+1] = w

p_acc = 1
t = 0

while t < 20:
    t += 1    
    p_acc_cal = 0
    # cum sum weights
    weight_cum = data_calculation[:int(N*alpha), num_var+1].cumsum(0)
    # normalization
    weight_cum = weight_cum/weight_cum[-1]
    
    for i in range(int(N*beta_N)):
        rand       = np.random.random_sample()
        particle   = np.sum(rand > weight_cum)
        data_resampling[i,:] = data_calculation[particle, 0:num_var+1]
    
    mu    = np.mean(data_calculation[:int(N*alpha), 0:num_var], 0)
    sigma = 1.1*np.var(data_calculation[:int(N*alpha), 0:num_var], 0)
    
    p_acc_cal = 0
    for i in range(int(N*beta_N), N):
        pho = 100
        while pho > kesi:
            p_acc_cal += 1
            theta       = np.random.multivariate_normal(mu, np.diag(sigma), 1)
            theta_input = (theta-min_para+2)/(max_para-min_para)
            feat_theta  = sess.run(feat_pred, feed_dict={para_input:theta_input})
            pho         = np.sum((feat_theta - feat_obser)**2)

        data_resampling[i, 0:num_var] = theta
        data_resampling[i, num_var]   = pho
    data_calculation[:, 0:num_var+1]  = data_resampling
    # sort the samples increasingly with pho  
    index = np.argsort(data_calculation[:, num_var])
    data_calculation = data_calculation[index]
    kesi = data_calculation[int(N*alpha), num_var]
    
    for i in range(int(N*alpha)):
        pho = data_calculation[i, num_var]
        w   = 1-(pho/kesi)**2
        data_calculation[i, num_var+1] = w
    p_acc = (N - N*beta_N)/ p_acc_cal
    std = np.std(data_calculation[:,0:num_var], 0)
    print('Iter '+str(t)+'_std:', std)
    print('kesi:', kesi)
    print('Accepted ratio: ', p_acc)
    P_acc_record.append(p_acc)
    Kesi_record.append(kesi)
    Appro_poster.append(data_calculation)

    save_name = 'ABC_NS_result'+str(noise)
    np.savez_compressed(save_name, a=Appro_poster, b=Kesi_record, c=P_acc_record)
