import sys
import lasagne
from lasagne import *
from lasagne.updates import *
from lasagne.nonlinearities import *
from lasagne.layers import *
use_dnn = True
try:
    from lasagne.layers.dnn import *
except:
    use_dnn = False
    sys.stderr.write("cudnn not detected\n")
if use_dnn:
    conv2d = Conv2DDNNLayer
    maxpool2d = MaxPool2DDNNLayer
else:
    conv2d = Conv2DLayer
    maxpool2d = MaxPool2DLayer
from lasagne.objectives import *
from lasagne.init import *
from nolearn.lasagne import *
import gzip
import os
import numpy as np
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
    flip_filters = False if "dont_flip_filters" in args else True
    max_epochs = args["max_epochs"]
    input_shape = args["input_shape"]
    kw = dict()
    l_in = InputLayer(input_shape)
    transform_prepool = maxpool2d(
        l_in,
        pool_size=(2,2))
    transform_conv1 = conv2d(transform_prepool,
        filter_size=(5,5),
        num_filters=20,
        flip_filters=flip_filters,
        nonlinearity=leaky_rectify)
    transform_pool1 = maxpool2d(transform_conv1,
        stride=2,
        pool_size=(2,2))
    transform_conv2 = conv2d(transform_pool1,
        filter_size=(5,5),
        num_filters=20,
        flip_filters=flip_filters,
        nonlinearity=leaky_rectify)
    transform_pool2 = maxpool2d(transform_conv2,
        stride=2,
        pool_size=(2,2))        
    transform_dense = DenseLayer(transform_pool2,
        num_units=20,
        nonlinearity=leaky_rectify)
    transform_six = DenseLayer(transform_dense,
        num_units=6,
        nonlinearity=identity)
    l_in = TransformerLayer(l_in, transform_six)
    l_prepool = maxpool2d(l_in, pool_size=(2,2) ) 
    l_conv1 = conv2d(l_prepool,
        num_filters=32,
        nonlinearity=leaky_rectify,
        flip_filters=flip_filters,
        filter_size=(5,5),
        stride=1)
    l_pool1 = maxpool2d(l_conv1,
        pool_size=(2,2),
        stride=2)
    l_conv2 = conv2d(l_pool1,
        num_filters=64,
        flip_filters=flip_filters,
        nonlinearity=leaky_rectify,
        filter_size=(3,3))
    l_pool2 = maxpool2d(l_conv2,
        pool_size=(2,2),
        stride=2)
    l_conv3 = conv2d(l_pool2,
        num_filters=64,
        flip_filters=flip_filters,
        nonlinearity=leaky_rectify,
        filter_size=(3,3))
    l_pool3 = maxpool2d(l_conv3,
        pool_size=(2,2),
        stride=2)
    # the paper did not mention having an fc before the softmax??
    #l_penult = DenseLayer(l_pool2,
    #    num_units=32,
    #    nonlinearity=leaky_rectify)
    l_out = DenseLayer(l_pool3,
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
    kw["batch_iterator_train"] = ShufflingBatchIterator(batch_size=bs)
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

