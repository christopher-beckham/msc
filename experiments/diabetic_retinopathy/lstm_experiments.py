
import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from lasagne.regularization import *
import sys
import numpy as np
from skimage import io, img_as_float
import cPickle as pickle
import os
from time import time
from keras.preprocessing.image import ImageDataGenerator
import gzip
from quadrant_network_dr import *

def integer_to_one_hot(X_train):
    #total = []
    total = np.zeros((X_train.shape[0], X_train[0].shape[0], 256))
    for b in range(0, X_train.shape[0]):
        #seqs = []
        seqs = np.zeros((X_train[b].shape[0], 256)).astype("float32")
        for i in range(0, X_train[b].shape[0]):
            one_hot = np.zeros((256)).astype("float32")
            one_hot[ X_train[b][i] ] = 1.0
            #seqs.append(one_hot)
            seqs[i] = one_hot
        #seqs = np.asarray(seqs, dtype="float32")
        #total.append(seqs)
        total[b] = seqs
    #total = np.asarray(total, dtype="float32")
    return total

if __name__ == '__main__':

    with gzip.open("../../data/mnist.pkl.gz") as f:
        dat = pickle.load(f)
    train_data, valid_data, test_data = dat
    X_train_I, _ = train_data
    X_valid_I, _ = valid_data
    
    X_train_I = (X_train_I*256).astype("int32")
    X_valid_I = (X_valid_I*256).astype("int32")

    def test():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm, {"batch_size": 128, "learning_rate":0.01, "num_units":20})
        train_lstm(cfg, out_file="output_lstm/test", num_epochs=10, debug=False, data=(X_train_I,X_valid_I))

    def test_klo():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm, {"batch_size": 128, "learning_rate":0.0001, "num_units":20, "klo":True})
        train_lstm(cfg, out_file="output_lstm/test", num_epochs=10, debug=False, data=(X_train_I,X_valid_I))
        
    def test_klo_orthog():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.0001, "num_units":200, "klo":True, "rmsprop":True})
        train_lstm(cfg, out_file="output_lstm/test_klo_orthog", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

        
        
    globals()[ sys.argv[1] ]()
