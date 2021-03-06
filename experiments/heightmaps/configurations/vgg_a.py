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

"""
This architecture is a subset of VGGNet A:
http://arxiv.org/pdf/1409.1556.pdf
"""
def get_net(args):
    if "relu" in args:
        nonlinearity=rectify
        sys.stderr.write("using relu\n")
    else:
        nonlinearity=tanh
    l_in = InputLayer( (None, 1, 256, 256) )
    l_noise = GaussianNoiseLayer(l_in, args["sigma"] if "sigma" in args else 0)
    #conv3@64
    l_conv1 = Conv2DLayer(
        l_noise, num_filters=64, filter_size=(3,3), nonlinearity=nonlinearity)
    #maxpool
    l_mp1 = MaxPool2DLayer(l_conv1, pool_size=(2,2))
    #conv3@128
    l_conv3 = Conv2DLayer(
        l_mp1, num_filters=128, filter_size=(3,3), nonlinearity=nonlinearity)
    #maxpool
    l_mp2 = MaxPool2DLayer(l_conv3, pool_size=(2,2))
    #conv3@256
    #conv3@256
    l_conv4 = Conv2DLayer(
        l_mp2, num_filters=256, filter_size=(3,3), nonlinearity=nonlinearity)
    l_conv5 = Conv2DLayer(
        l_conv4, num_filters=256, filter_size=(3,3), nonlinearity=nonlinearity)
    #maxpool
    l_mp3 = MaxPool2DLayer(l_conv5, pool_size=(2,2))
    #conv3@512
    #conv3@512
    l_conv6 = Conv2DLayer(
        l_mp3, num_filters=512, filter_size=(3,3), nonlinearity=nonlinearity)
    l_conv7 = Conv2DLayer(
        l_conv6, num_filters=512, filter_size=(3,3), nonlinearity=nonlinearity)
    l_mp4 = MaxPool2DLayer(l_conv7, pool_size=(2,2))
    l_conv8 = Conv2DLayer(
        l_mp4, num_filters=512, filter_size=(3,3), nonlinearity=nonlinearity)
    l_conv9 = Conv2DLayer(
        l_conv8, num_filters=512, filter_size=(3,3), nonlinearity=nonlinearity)
    l_out = l_conv9
    for layer in get_all_layers(l_out)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_out = InverseLayer(l_out, layer)
    l_out = NonlinearityLayer(l_out, nonlinearity=sigmoid)
    sys.stderr.write( "number of params: %i\n" % count_params(l_out) )
    for layer in get_all_layers(l_out):
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    return {
        "l_out": l_out,
        "target_layers": {
            "l_conv1": l_conv1,
            "l_conv3": l_conv3,
            "l_conv4": l_conv4,
            "l_conv5": l_conv5,
            "l_conv6": l_conv6,
            "l_conv7": l_conv7,
            "l_conv8": l_conv8,
            "l_conv9": l_conv9
        },
        "use_rgb":False
    }

if __name__ == "__main__":
    print "debugging..."
    get_net({})
