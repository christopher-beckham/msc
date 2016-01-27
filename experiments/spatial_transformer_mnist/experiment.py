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
    input_shape = args["input_shape"]
    kw = dict()
    l_in = InputLayer(input_shape)
    transform_prepool = MaxPool2DLayer(
        l_in,
        pool_size=(2,2))
    transform_conv1 = Conv2DLayer(transform_prepool,
        filter_size=(5,5),
        num_filters=20,
        nonlinearity=leaky_rectify)
    transform_pool1 = MaxPool2DLayer(transform_conv1,
        stride=1,
        pool_size=(2,2))
    transform_conv2 = Conv2DLayer(transform_pool1,
        filter_size=(5,5),
        num_filters=20,
        nonlinearity=leaky_rectify)
    transform_pool2 = MaxPool2DLayer(transform_conv2,
        stride=1,
        pool_size=(2,2))        
    transform_dense = DenseLayer(transform_pool2,
        num_units=20,
        nonlinearity=leaky_rectify)
    transform_six = DenseLayer(transform_dense,
        num_units=6,
        nonlinearity=identity)
    l_in = TransformerLayer(l_in, transform_six)
    l_prepool = MaxPool2DLayer(l_in, pool_size=(2,2)) 
    l_conv1 = Conv2DLayer(l_prepool,
        num_filters=16,
        nonlinearity=leaky_rectify,
        filter_size=(9,9),
        stride=1)
    l_pool1 = MaxPool2DLayer(l_conv1,
        pool_size=(2,2),
        stride=2)
    l_conv2 = Conv2DLayer(l_pool1,
        num_filters=32,
        nonlinearity=leaky_rectify,
        filter_size=(7,7))
    l_pool2 = MaxPool2DLayer(l_conv2,
        pool_size=(2,2),
        stride=2)
    l_out = DenseLayer(l_pool2,
        num_units=10,
        nonlinearity=softmax)
    kw["max_epochs"] = max_epochs
    kw["update"] = adagrad
    kw["update_learning_rate"] = args["alpha"]
    kw["verbose"] = 1
    kw["eval_size"] = 0.1
    out_model = args["out_model"]
    out_stats = args["out_stats"]
    kw["on_epoch_finished"] = [ save_model_on_best(out_model), save_stats_on_best(out_stats) ]
    bs = args["batch_size"]
    kw["batch_iterator_train"] = BatchIterator(batch_size=bs)
    net = NeuralNet(l_out, **kw)
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

