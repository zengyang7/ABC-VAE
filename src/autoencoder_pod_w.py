#!/usr/bin/env python3
""" Training the numeric data """

# standard library imports
import os, time, sys
import scipy.io as scio
from scipy.stats import multivariate_normal as mvn

# third party imports
import numpy as np
import tensorflow as tf


tf.reset_default_graph()

# read inputs from inputs file
# print(sys.argv[1])
file_para = open(sys.argv[1], 'r')
#file_name1 = '/Users/zengyang/VAE/demo/6_nonlinear/setting'
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
    exec("%s = %.3f" % (var_name, var_value))
num = int(num)
N = int(N)

print('The size of training samples: ', str(training_size))
## load data
mat_file = scio.loadmat(sys.argv[2])
# test code
#mat_file_path = '/Users/zengyang/VAE/demo/6_nonlinear/sensitive_data_6_10000.mat'
#mat_file = scio.loadmat(mat_file_path)
training_size = 8000

parameters = mat_file['parameter_space']
temperature = mat_file['T_sensitive'].T

parameters = parameters/nor_par
# normalization
t_data = (temperature-np.min(temperature)+1)/(1.2*(np.max(temperature)-np.min(temperature)))

train_data = t_data[0:training_size]
test_data = t_data[training_size:-1]
train_para = parameters[0:training_size]
test_para = parameters[training_size:-1]
#train_data = temperature/np.max(temperature)


## setting
# number of basis vector of PCA and autoencoder
#num = 2

# learning rate of autoencoder
learning_rate1 = 0.0001

# learning rate of NN from parameters to reduced coefficients
learning_rate2 = 0.001
beta = 0.9

# batch size
batch_size = 64

# epoch for traing autoencoder
epoch1 = 20000

# epoch for training NN from parameters to reduced coefficients
epoch2 = 20000

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

## PCA
mu = train_data.mean(axis=0)
U,s,V = np.linalg.svd(train_data-mu, full_matrices=False)
Zpca = np.dot(test_data - mu, V.transpose())

Rpca = np.dot(Zpca[:,:num],V[:num, :])+mu   # reconstruction
Pred_pca = Rpca*1.2*(np.max(temperature)-np.min(temperature))+np.min(temperature)-1
err = np.sum((Pred_pca-temperature[training_size:-1])**2)/Rpca.shape[0]/Rpca.shape[1]
R_s_pca = R_squared(Pred_pca, temperature[training_size:-1])
print('PCA reconstruction error with ' + str(num)+ ' PCs:'+str(round(err, 5)))
print('R square of PCA with '+ str(num)+ ' PCs:'+str(round(R_s_pca, 5)))

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
    x = lrelu(tf.layers.batch_normalization(x))
    return x

# placeholder for Temperature field
T_input = tf.placeholder("float", [None, temperature.shape[1]])
# placeholder for boundary condition
para_input = tf.placeholder("float", [None, parameters.shape[1]])

encoder_variable = Encoder(T_input)
T_Pred = Decoder(encoder_variable)

# loss function of autoencoder
loss_vae = tf.reduce_mean(tf.pow(T_input-T_Pred, 2))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer_vae = tf.train.AdamOptimizer(learning_rate1, beta).minimize(loss_vae, var_list=theta)

# relationship between boundary parameters and reduced coefficient
encoder_pred = Para2Enc(para_input)
T_P_Pred = Decoder(encoder_pred)
loss_para2enc = tf.reduce_sum(tf.pow(encoder_variable-encoder_pred, 2))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optimizer_para2enc = tf.train.AdamOptimizer(learning_rate2,beta).minimize(loss_para2enc, var_list=theta_P)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1, epoch1+1):
    batch_x, _ = next_batch(batch_size, train_para, train_data)
    _, l = sess.run([optimizer_vae, loss_vae], feed_dict={T_input:batch_x})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l))
        
# Testing
# The prediction of autoencoder
Ze_vae, Re_vae = sess.run([encoder_variable, T_Pred], feed_dict={T_input:test_data})

# inverse of normalization
Pred_vae = Re_vae*(np.max(temperature)-np.min(temperature))*1.2+np.min(temperature)-1

# average of error
err_vae = np.sum((temperature[training_size:-1]-Pred_vae)**2)/Re_vae.shape[0]/Re_vae.shape[1]
print('VAE reconstruction error with ' + str(num)+ ' PCs:'+str(round(err_vae, 5)))
delta = Pred_vae-temperature[training_size:-1]
delta_percent = delta/temperature[training_size:-1]
print(np.max(np.abs(delta_percent)))

# R square
R_s_ae = R_squared(Pred_vae, temperature[training_size:-1])
print('R square of ae with ' + str(num)+ ' PCs:'+str(round(R_s_ae, 5)))

for i in range(1, epoch2+1):
    batch_x, batch_y = next_batch(batch_size, train_para, train_data)
    _, l1 = sess.run([optimizer_para2enc, loss_para2enc], 
                     feed_dict={T_input:batch_x, para_input:batch_y})
    if i % 1000 == 0 or i==1:
        print('Step %i: Minibatch Loss: %f' % (i, l1))

# Prediction of NN
Ze_vae_s, Re_vae_s = sess.run([encoder_pred, T_P_Pred], feed_dict={para_input:test_para})

# inverse of normalization
Pred_vae_s = Re_vae_s*(np.max(temperature)-np.min(temperature))*1.2+np.min(temperature)-1


# average of error
err_vae_s = np.sum((temperature[training_size:-1]-Pred_vae_s)**2)/Re_vae_s.shape[0]/Re_vae_s.shape[1]
print('VAE reconstruction error with ' + str(num)+ ' PCs:'+str(round(err_vae_s, 5)))
delta = Pred_vae_s-temperature[training_size:-1]
delta_percent = delta/temperature[training_size:-1]
print(np.max(np.abs(delta_percent)))

# R square
R_s_ae_s = R_squared(Pred_vae_s, temperature[training_size:-1])
print('R square of ae with ' + str(num)+ ' PCs:'+str(round(R_s_ae_s, 5)))


###############################################################################
# PMC
###############################################################################
#observationname = '/Users/zengyang/VAE/demo/6_nonlinear/observation_data.mat'
Observations_file = scio.loadmat(sys.argv[3])
#Observations_file = scio.loadmat(observationname)
Observations = Observations_file['T0'].T
Observations = (Observations-np.min(temperature)+1)/(1.2*(np.max(temperature)-np.min(temperature)))

Ze_observation = sess.run(encoder_variable, feed_dict={T_input:Observations})

# prior
mu_prior = np.array([900, 2000, 1340, 1060, 2100, 1300])
sigma_prior = np.diag((mu_prior*0.05)**2)

Kesi_record  = []
Sigma_record = []
Appro_poster = []
P_acc_record = []

data_calculation = np.zeros([N, parameters.shape[1]+2])

for i in range(N):
    theta   = np.random.multivariate_normal(mu_prior, sigma_prior, 1)/nor_par
    Ze_pred = sess.run(encoder_pred, feed_dict={para_input:theta})
    pho     = np.sum((Ze_pred - Ze_observation)**2)
    w = 1
    data_calculation[i, 0:6] = theta*nor_par
    data_calculation[i, 6]   = w/N
    data_calculation[i, 7]   = pho
    
index = np.argsort(data_calculation[:,-1])
data_calculation = data_calculation[index]

kesi  = data_calculation[int(N*alpha), -1]
sigma = 2 * np.var(data_calculation[0:int(N*alpha),0:6], 0)

Kesi_record.append(kesi)
Sigma_record.append(sigma)
Appro_poster.append(data_calculation)

p_acc = 1
t = 1

while p_acc > p_acc_min:
    t += 1
    if t >= 40:
        break
    print(t)
    p_acc_cal = 0
    
    for i in range(int(N*alpha), N):
        
        ##sample from last particle based on the weight
        # cum sum weights
        weight_cum = data_calculation[:int(N*alpha),6].cumsum(0)
        # normalization
        weight_cum = weight_cum/weight_cum[-1]
        # generata random value
        rand     = np.random.random_sample()
        particle = np.sum(rand > weight_cum)
        theta_particle = data_calculation[particle, 0:6]
        
        # generate new sample with the particle
        theta_new = np.random.multivariate_normal(theta_particle, np.diag(sigma), 1)
        
        theta_nn_input = theta_new/nor_par
        
        # determine the weight of new sample
        Ze_pred = sess.run(encoder_pred, feed_dict={para_input:theta_nn_input})
        pho     = np.sum((Ze_pred - Ze_observation)**2)   
        sum_w   = 0
        
        for j in range(int(N*alpha)):
            w_j       = weight_cum[j]
            # N(theta_new, theta^(t-1)_j, sigma)
            pdf_theta = mvn.pdf(theta_new, mean=data_calculation[j, 0:6], cov=np.diag(sigma))
            sum_w    += w_j*pdf_theta
        w = mvn.pdf(theta_new, mean=mu_prior, cov=sigma_prior)/sum_w
        
        # new samples
        data_calculation[i, 0:6] = theta_new*nor_par
        data_calculation[i, 6] = w
        data_calculation[i, 7] = pho
        if pho<kesi:
            p_acc_cal += 1
    p_acc = p_acc_cal/(N - N*alpha)
    print(p_acc)
    index = np.argsort(data_calculation[:,-1])
    data_calculation = data_calculation[index]
    
    kesi = data_calculation[int(N*alpha), -1]
    std = np.std*(data_calculation[0:int(N*alpha),0:6], 0)
    print('Iter'+str(t)+'_std:', std)
    sigma = 2 * np.var(data_calculation[0:int(N*alpha),0:6], 0)
    
    P_acc_record.append(p_acc)
    Kesi_record.append(kesi)
    Sigma_record.append(sigma)
    Appro_poster.append(data_calculation)

    save_name = 'ABC_results'
    np.savez_compressed(save_name, a=Appro_poster, b=Kesi_record)
