import lasagne
from lasagne import *
from lasagne.updates import *
from lasagne.nonlinearities import *
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.init import *
from nolearn.lasagne import *
import gzip
import os
import numpy as np
import sys
import re
import random
import tempfile
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
sys.path.append("../../modules/")
from helper import *
    
def get_net(args):
    max_epochs = args["max_epochs"]
    kw = dict()
    layer_conf = [
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('dense', layers.DenseLayer),
        ('output', layers.DenseLayer)
    ]
    kw["input_shape"] = (None, 1, 28, 28)
    kw["conv1_num_filters"] = 16
    kw["conv1_filter_size"] = (5,5)
    kw["conv1_W"] = GlorotUniform(gain="relu")
    kw["conv1_nonlinearity"] = leaky_rectify
    kw["pool1_pool_size"] = (2,2)
    kw["conv2_num_filters"] = 32
    kw["conv2_filter_size"] = (3,3)
    kw["conv2_W"] = GlorotUniform(gain="relu")
    kw["conv2_nonlinearity"] = leaky_rectify
    kw["pool2_pool_size"] = (2,2)
    kw["conv3_num_filters"] = 32
    kw["conv3_filter_size"] = (3,3)
    kw["conv3_W"] = GlorotUniform(gain="relu")
    #kw["pool3_pool_size"] = (2,2)
    kw["dense_num_units"] = 100
    kw["dense_nonlinearity"] = leaky_rectify
    kw["dense_W"] = GlorotUniform(gain="relu")
    kw["output_num_units"] = 10
    kw["output_nonlinearity"] = softmax
    kw["output_W"] = GlorotUniform()
    kw["layers"] = layer_conf
    kw["max_epochs"] = max_epochs
    kw["update"] = adagrad
    kw["update_learning_rate"] = args["alpha"]
    kw["verbose"] = 1
    kw["eval_size"] = 0.1
    out_model = args["out_model"]
    out_stats = args["out_stats"]
    kw["on_epoch_finished"] = [ save_model_on_best(out_model), save_stats_on_best(out_stats) ]
    net = NeuralNet(**kw)
    return net
    
def train(args):
    np.random.seed( args["seed"] )
    random.seed( args["seed"] )
    y_train = np.asarray(args["y_train"], dtype="int32")
    net1 = get_net(args)
    net1.initialize()
    X_train = np.asarray(args["X_train"], dtype="float32")
    #with Capturing() as output:
    model = net1.fit(X_train, y_train)
    #return { "results": remove_colour("\n".join(output)),
    #    "params": net1.get_all_params_values() }

def describe(args, model):
    return model["results"]

def test(args, model):
    net1 = get_net(args)
    net1.load_params_from(model["params"])
    X_test = np.asarray(args["X_test"], dtype="float32")
    X_test = X_test.reshape( (X_test.shape[0], 1, X_test.shape[1]) )
    return net1.predict_proba(X_test).tolist()

if __name__ == "__main__":
    data = load_mnist("../../data/mnist.pkl.gz")
    train_set, _, _ = data
    args = dict()
    args["X_train"] = train_set[0]
    args["y_train"] = train_set[1]
    args["max_epochs"] = 100
    args["alpha"] = 0.01
    args["seed"] = 0
    args["out_model"] = "exp1.model"
    args["out_stats"] = "exp1.stats"
    train(args)
    
    print data
