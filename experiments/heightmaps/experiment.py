
import numpy as np
import sys
import os
import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
import random
from time import time
import train_ae
import cPickle as pickle

def exp1():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    #x_filename = "train_data_minimal.npy"
    X_train = np.load(x_filename).astype("float32")
    sys.stderr.write("X_train shape: %s\n" % str(X_train.shape))
    args = dict()
    args["X_train"] = X_train
    #args["rmsprop"] = True
    args["num_epochs"] = 30
    args["learning_rate"] = 0.01
    args["batch_size"] = 128
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["in_pkl"] = "output/model.pkl"
    args["config"] = "basic.py"
    train_ae.train(args)

def exp2():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    #x_filename = "train_data_minimal.npy"
    X_train = np.load(x_filename).astype("float32")
    sys.stderr.write("X_train shape: %s\n" % str(X_train.shape))
    args = dict()
    args["X_train"] = X_train
    #args["rmsprop"] = True
    args["num_epochs"] = 30
    args["learning_rate"] = 0.01
    args["batch_size"] = 128
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    #args["in_pkl"] =
    args["config"] = "basic2.py"
    train_ae.train(args)

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    #exp1()
    exp2()