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
import cPickle as pickle
import os
from time import time
import gzip
from quadrant_network_dr import *

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

    def test_klo_orthog_lr0001():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.0001, "num_units":200, "mode":"klo", "rmsprop":True})
        train_lstm(cfg, out_file="output_lstm/test_klo_orthog_lr0.001", num_epochs=30, debug=False, data=(X_train_I,X_valid_I),resume="/data/lisatmp4/beckhamc/models_neat/test_klo_orthog_lr0.001.modelv2.30.bak")
        
    def test_xent_orthog():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.0001, "num_units":200, "rmsprop":True})
        train_lstm(cfg, out_file="output_lstm/test_xent_orthog", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

    def test_xent_orthog_lr001():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.01, "num_units":200, "rmsprop":True})
        train_lstm(cfg, out_file="output_lstm/test_xent_orthog_lr0.01", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

    def test_qwk_orthog_lr0001():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":200, "mode":"qwk", "rmsprop":True})
        train_lstm(cfg, out_file="output_lstm/test_qwk_orthog_lr0.001", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

    # ----------------------

    def test_xent_orthog_100u_clip1():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":100, "optim":"rmsprop", "mode":"xent", "clip":1.0})
        train_lstm(cfg, out_file="output_lstm/new_test_xent_orthog_100u_lr0.001_clip1", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

    def test_xent_orthog_100u_clip1_forget1():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":100, "optim":"rmsprop", "mode":"xent", "clip":1.0, "forget_init":1.0})
        train_lstm(cfg, out_file="output_lstm/new_test_xent_orthog_100u_lr0.001_clip1_forget1", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))
        
    def test_xent_orthog_200u_clip1_forget1():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":200, "optim":"rmsprop", "mode":"xent", "clip":1.0, "forget_init":1.0})
        train_lstm(cfg, out_file="output_lstm/new_test_xent_orthog_200u_lr0.001_clip1_forget1", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

    def test_klo_orthog_200u_clip1_forget1():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":200, "optim":"rmsprop", "mode":"klo", "clip":1.0, "forget_init":1.0})
        train_lstm(cfg, out_file="output_lstm/new_test_klo_orthog_200u_lr0.001_clip1_forget1", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

    def test_qwk_orthog_200u_clip1_forget1():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":200, "optim":"rmsprop", "mode":"qwk", "clip":1.0, "forget_init":1.0})
        train_lstm(cfg, out_file="output_lstm/new_test_qwk_orthog_200u_lr0.001_clip1_forget1", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))


        
    def test_xent_orthog_300u_clip1_forget1():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":300, "optim":"rmsprop", "mode":"xent", "clip":1.0, "forget_init":1.0})
        train_lstm(cfg, out_file="output_lstm/new_test_xent_orthog_300u_lr0.001_clip1_forget1", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))

    def test_xent_orthog_400u_clip1_forget1():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":400, "optim":"rmsprop", "mode":"xent", "clip":1.0, "forget_init":1.0})
        train_lstm(cfg, out_file="output_lstm/new_test_xent_orthog_400u_lr0.001_clip1_forget1", num_epochs=30, debug=False, data=(X_train_I,X_valid_I))
        
        
    def test_xent_orthog_200u_clip1_forget5():
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_lstm(lstm_orthog, {"batch_size": 128, "learning_rate":0.001, "num_units":200, "optim":"rmsprop", "mode":"xent", "clip":1.0, "forget_init":5.0})
        train_lstm(cfg, out_file="output_lstm/new_test_xent_orthog_200u_lr0.001_clip1_forget5", num_epochs=50, debug=False, data=(X_train_I,X_valid_I))

        
    globals()[ sys.argv[1] ]()
