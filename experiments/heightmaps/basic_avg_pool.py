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
import cPickle as pickle

# basic1
def get_net(args):
    l_in = InputLayer( (None, 1, 256, 256) )
    l_noise = GaussianNoiseLayer(l_in, args["sigma"] if "sigma" in args else 0)
    l_conv1 = Conv2DLayer(
        l_noise, num_filters=32, filter_size=(3,3), nonlinearity=tanh)
    l_mp1 = Pool2DLayer(l_conv1, pool_size=(2,2), mode="average_inc_pad")
    l_conv2 = Conv2DLayer(
        l_mp1, num_filters=64, filter_size=(3,3), nonlinearity=tanh)
    l_mp2 = Pool2DLayer(l_conv2, pool_size=(2,2), mode="average_inc_pad")
    l_conv3 = Conv2DLayer(
        l_mp2, num_filters=128, filter_size=(3,3), nonlinearity=tanh)
    l_conv4 = Conv2DLayer(
        l_conv3, num_filters=128, filter_size=(3,3), nonlinearity=tanh)
    l_out = l_conv4
    for layer in get_all_layers(l_out)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_out = InverseLayer(l_out, layer)
    l_out = NonlinearityLayer(l_out, nonlinearity=sigmoid)
    sys.stderr.write( "number of params: %i\n" % count_params(l_out) )
    for layer in get_all_layers(l_out):
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    return l_out

if __name__ == "__main__":
    print "debugging..."
    get_net({})