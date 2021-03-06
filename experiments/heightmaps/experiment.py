
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

def exp():
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
    sys.stderr.write("for exp2 using batch size of 32 instead of 128...\n")
    args["batch_size"] = 32
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    #args["in_pkl"] =
    args["config"] = "basic2.py"
    train_ae.train(args)

def exp3():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_train = np.load(x_filename).astype("float32")
    sys.stderr.write("X_train shape: %s\n" % str(X_train.shape))
    args = dict()
    args["X_train"] = X_train
    args["num_epochs"] = 50
    args["learning_rate"] = 0.01
    sys.stderr.write("for exp2 using batch size of 64 instead of 128...\n")
    args["batch_size"] = 64
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    #args["in_pkl"] =
    args["config"] = "basic_avg_pool.py"
    train_ae.train(args)

def exp4():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_train = np.load(x_filename).astype("float32")
    sys.stderr.write("X_train shape: %s\n" % str(X_train.shape))
    args = dict()
    args["X_train"] = X_train
    args["num_epochs"] = 100
    args["learning_rate"] = 0.01
    #sys.stderr.write("for exp2 using batch size of 64 instead of 128...\n")
    args["batch_size"] = 64
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["in_pkl"] = "output/vgg_a_subset.pkl"
    args["config"] = "vgg_a_subset.py"
    train_ae.train(args)

def exp5():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    sys.stderr.write("X_all shape: %s\n" % str(X_all.shape))
    args = dict()
    args["X_all"] = X_all
    args["num_epochs"] = 100
    args["learning_rate"] = 0.01
    #sys.stderr.write("for exp2 using batch size of 64 instead of 128...\n")
    args["batch_size"] = 64
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a_subset.py"
    train_ae.train(args)

def exp6():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    sys.stderr.write("X_all shape: %s\n" % str(X_all.shape))
    args = dict()
    args["X_all"] = X_all
    args["num_epochs"] = 100
    args["learning_rate"] = 0.01
    #sys.stderr.write("for exp2 using batch size of 64 instead of 128...\n")
    args["batch_size"] = 32
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a_subset_2.py"
    train_ae.train(args)

def exp7():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    sys.stderr.write("X_all shape: %s\n" % str(X_all.shape))
    args = dict()
    args["X_all"] = X_all
    args["num_epochs"] = 100
    args["learning_rate"] = 0.01
    #sys.stderr.write("for exp2 using batch size of 64 instead of 128...\n")
    args["batch_size"] = 32
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a_subset_beefed.py"
    train_ae.train(args)

def exp8():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    sys.stderr.write("X_all shape: %s\n" % str(X_all.shape))
    args = dict()
    args["X_all"] = X_all
    args["num_epochs"] = 100
    args["learning_rate"] = 0.01
    #sys.stderr.write("for exp2 using batch size of 64 instead of 128...\n")
    args["batch_size"] = 32
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a_subset_less_depth.py"
    train_ae.train(args)

def exp9():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    sys.stderr.write("X_all shape: %s\n" % str(X_all.shape))
    args = dict()
    args["X_all"] = X_all
    args["num_epochs"] = 100
    args["learning_rate"] = 0.01
    #sys.stderr.write("for exp2 using batch size of 64 instead of 128...\n")
    args["batch_size"] = 32
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a_subset_less_depth_relu.py"
    args["norm_constraint"] = 1
    train_ae.train(args)

def exp10():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    X_train = X_all[0 : 0.9*X_all.shape[0]]
    X_valid = X_all[0.9*X_all.shape[0] ::]
    sys.stderr.write("X_all shape: %s\n" % str(X_all.shape))
    args = dict()
    args["X_train"] = X_train
    args["X_valid"] = X_valid
    args["num_epochs"] = 100
    args["learning_rate"] = 0.01
    args["batch_size"] = 48
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a.py"
    train_ae.train(args)

def exp8_only_one_example():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    X_train = X_all[5:6]
    X_valid = np.asarray([], dtype="float32")
    args = dict()
    args["X_train"] = X_train
    args["X_valid"] = X_valid
    args["num_epochs"] = 100000
    args["learning_rate"] = 0.001 # was 0.01 for first iteration
    args["batch_size"] = 1
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a_subset_less_depth.py"
    args["in_pkl"] = "output/only_one_example.pkl"
    train_ae.train(args)

def exp10_only_one_example():
    out_pkl = sys.argv[1]
    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_all = np.load(x_filename).astype("float32")
    X_train = X_all[5:6]
    X_valid = np.asarray([], dtype="float32")
    args = dict()
    args["X_train"] = X_train
    args["X_valid"] = X_valid
    args["num_epochs"] = 20000
    args["learning_rate"] = 0.01
    args["batch_size"] = 1
    args["momentum"] = 0.9
    args["out_pkl"] = out_pkl
    args["config"] = "configurations/vgg_a.py"
    train_ae.train(args)

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    #exp1()
    #exp2()
    #exp3()
    #exp4()
    #exp5()
    #exp6()
    #exp7()
    #exp8()
    #exp9()
    #exp10()
    exp8_only_one_example()
    #exp10_only_one_example()
