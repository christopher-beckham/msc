import sys
import lasagne
from lasagne import *
from lasagne.updates import *
from lasagne.nonlinearities import *
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.init import *
from nolearn.lasagne import *
use_dnn = True
try:
    from lasagne.layers.dnn import *
except:
    sys.stderr.write("cudnn not detected, using regular conv/mp classes\n")
    use_dnn = False
if use_dnn:
    conv2d = Conv2DDNNLayer
    maxpool2d = MaxPool2DDNNLayer
else:
    conv2d = Conv2DLayer
    maxpool2d = MaxPool2DLayer
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
    max_epochs = args["max_epochs"]
    #input_shape = args["input_shape"]
    kw = dict()
    l_in = lasagne.layers.InputLayer(
        shape=(None, 3, 256, 256),
    )
    # { "type": "CONV", "num_filters": 32, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" },
    l_conv1 = conv2d(
        l_in,
        num_filters=32,
        filter_size=(3,3),
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    l_pool1 = maxpool2d(
        l_conv1,
        pool_size=(3,3),
        stride=2
    )
    # { "type": "CONV", "dropout": 0.1, "num_filters": 64, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" }
    l_dropout1 = lasagne.layers.DropoutLayer(l_pool1, p=0.1)
    l_conv2 = conv2d(
        l_dropout1,
        num_filters=64,
        filter_size=(3,3),
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    l_pool2 = maxpool2d(
        l_conv2,
        pool_size=(3,3),
        stride=2
    )
    # { "type": "CONV", "dropout": 0.1, "num_filters": 128, "filter_size": 3, "nonlinearity": "LReLU" },
    l_dropout2 = lasagne.layers.DropoutLayer(l_pool2, p=0.1)
    l_conv3 = conv2d(
        l_dropout2,
        num_filters=128,
        filter_size=(3,3),
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    # { "type": "CONV", "dropout": 0.1, "num_filters": 128, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" },
    l_dropout3 = lasagne.layers.DropoutLayer(l_conv3, p=0.1)
    l_conv4 = conv2d(
        l_dropout3,
        num_filters=128,
        filter_size=(3,3),
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    l_pool3 = maxpool2d(
        l_conv4,
        pool_size=(3,3),
        stride=2
    )
    # { "type": "CONV", "dropout": 0.1, "num_filters": 128, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" },
    l_dropout4 = lasagne.layers.DropoutLayer(l_pool3, p=0.1)
    l_conv5 = conv2d(
        l_dropout4,
        num_filters=128,
        filter_size=(3,3),
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    # maxpool size 3 stride 2
    l_pool4 = maxpool2d(
        l_conv5,
        pool_size=(3,3),
        stride=2
    )
    # { "type": "CONV", "dropout": 0.1, "num_filters": 256, "filter_size": 3, "pool_size": 2, "pool_stride": 2, "nonlinearity": "LReLU" },
    l_dropout5 = lasagne.layers.DropoutLayer(l_pool4, p=0.1)
    l_conv6 = conv2d(
        l_dropout5,
        num_filters=256,
        filter_size=(3,3),
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    l_pool5 = maxpool2d(
        l_conv6,
        pool_size=(2,2),
        stride=2
    )
    # { "type": "FC", "dropout": 0.5, "num_units": 2048, "pool_size": 2, "nonlinearity": "LReLU" },
    l_dropout6 = lasagne.layers.DropoutLayer(l_pool5, p=0.5)
    l_hidden1 = lasagne.layers.DenseLayer(
        l_dropout6,
        num_units=2048,
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    l_pool6 = lasagne.layers.FeaturePoolLayer(
        l_hidden1,
        pool_size=2
    )
    # { "type": "FC", "dropout": 0.5, "num_units": 2048, "pool_size": 2, "nonlinearity": "LReLU" },
    l_dropout7 = lasagne.layers.DropoutLayer(l_pool6, p=0.5)
    l_hidden2 = lasagne.layers.DenseLayer(
        l_dropout7,
        num_units=2048,
        nonlinearity=leaky_rectify,
        W=GlorotUniform()
    )
    l_pool7 = lasagne.layers.FeaturePoolLayer(
        l_hidden2,
        pool_size=2
    )
    # { "type": "OUTPUT", "dropout": 0.5, "nonlinearity": "sigmoid" }
    l_dropout8 = lasagne.layers.DropoutLayer(l_pool7, p=0.5)
    l_out = lasagne.layers.DenseLayer(
        l_dropout8,
        num_units=5,
        nonlinearity=softmax,
        W=GlorotUniform()
    )
    kw["objective_loss_function"] = get_kappa_loss(5)
    kw["max_epochs"] = max_epochs
    kw["update"] = nesterov_momentum
    kw["update_learning_rate"] = args["alpha"]
    kw["verbose"] = 1
    kw["train_split"] = TrainSplit(eval_size=0.15)
    out_model = args["out_model"]
    out_stats = args["out_stats"]
    kw["on_epoch_finished"] = [ save_model_on_best(out_model), save_stats_on_best(out_stats) ]
    kw["custom_score"] = ["valid_kappa", np_kappa]
    bs = args["batch_size"]
    filenames = args["filenames"]
    prefix = args["prefix"]
    kw["batch_iterator_train"] = ImageBatchIterator(
        batch_size=bs, shuffle=True, filenames=filenames, prefix=prefix, zmuv=True, augment=True)
    kw["batch_iterator_test"] = ImageBatchIterator(
        batch_size=bs, shuffle=False, filenames=filenames, prefix=prefix, zmuv=True, augment=False)
    net = NeuralNet(l_out, **kw)
    return net
    
def train(args):
    np.random.seed( args["seed"] )
    random.seed( args["seed"] )
    y_train = np.asarray(args["y_train"], dtype="int32")
    net1 = get_net(args)
    if "in_model" in args:
        sys.stderr.write("loading existing model: %s\n" % args["in_model"])
        net1.load_params_from(args["in_model"])
    else:
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

