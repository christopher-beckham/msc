
# coding: utf-8

# In[149]:

import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.regularization import *
from lasagne.random import get_rng
from lasagne.updates import *
from lasagne.init import *
import numpy as np
import sys
sys.path.append("../../modules/")
import helper as hp

import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

import os
import cPickle as pickle

import deep_residual_learning_CIFAR10

import math


# In[83]:

data = deep_residual_learning_CIFAR10.load_data()
sys.stderr.write("loading smaller version of cifar10...\n")
X_train_and_valid, y_train_and_valid, X_test, y_test =     data["X_train"], data["Y_train"], data["X_test"], data["Y_test"]
X_train = X_train_and_valid[ 0 : 0.9*X_train_and_valid.shape[0] ]
y_train = y_train_and_valid[ 0 : 0.9*y_train_and_valid.shape[0] ]
X_valid = X_train_and_valid[ 0.9*X_train_and_valid.shape[0] :: ]
y_valid = y_train_and_valid[ 0.9*y_train_and_valid.shape[0] :: ]


# In[152]:

X_train_flip = X_train[:,:,:,::-1]
y_train_flip = y_train
X_train = np.concatenate((X_train,X_train_flip),axis=0)
y_train = np.concatenate((y_train,y_train_flip),axis=0)


# In[102]:

def disp_form(img):
    return np.transpose(img, [2,1,0])


# In[126]:

new_X_train = []
for img in X_train:
    tmp = np.zeros((3,40,40))
    tmp[:, 4:-4, 4:-4] = img
    new_X_train.append(tmp)

new_X_valid = []
for img in X_valid:
    tmp = np.zeros((3,40,40))
    tmp[:, 4:-4, 4:-4] = img
    new_X_valid.append(tmp)
    
new_X_test = []
for img in X_test:
    tmp = np.zeros((3,40,40))
    tmp[:, 4:-4, 4:-4] = img
    new_X_test.append(tmp)    

new_X_train = np.asarray(new_X_train)
print new_X_train.shape

img = new_X_train[8]

def extract_crop(img):
    rand_x, rand_y = np.random.randint(0,9), np.random.randint(0,9)
    return img[:,rand_x:rand_x+32,rand_y:rand_y+32]

np.savez("cifar10.npz", 
    X_train=new_X_train, 
    y_train=y_train, 
    X_valid=X_valid, 
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test
)

print new_X_train.shape, y_train.shape, new_X_valid.shape, y_valid.shape, new_X_test.shape, y_test.shape



