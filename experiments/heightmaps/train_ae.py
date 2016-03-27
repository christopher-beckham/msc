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

def get_net(args):
    l_in = InputLayer( (None, 1, 256, 256) )
    l_noise = GaussianNoiseLayer(l_in, args["sigma"] if "sigma" in args else 0)
    l_conv1 = Conv2DLayer(
        l_noise, num_filters=32, filter_size=(3,3), nonlinearity=tanh)
    l_mp1 = MaxPool2DLayer(l_conv1, pool_size=(2,2))
    #l_drop1 = DropoutLayer(l_mp1, p=p)
    l_conv2 = Conv2DLayer(
        l_mp1, num_filters=64, filter_size=(3,3), nonlinearity=tanh)
    l_mp2 = MaxPool2DLayer(l_conv2, pool_size=(2,2))
    #l_drop2 = DropoutLayer(l_mp2, p=p)
    l_conv3 = Conv2DLayer(
        l_mp2, num_filters=128, filter_size=(3,3), nonlinearity=tanh)
    #l_drop3 = DropoutLayer(l_conv3, p=p)
    l_conv4 = Conv2DLayer(
        l_conv3, num_filters=128, filter_size=(3,3), nonlinearity=tanh)
    #l_mp3 = MaxPool2DLayer(l_conv4, pool_size=(3,3))
    #l_drop4 = DropoutLayer(l_mp3, p=p)
    #l_conv5 = Conv2DLayer(
    #    l_mp3, num_filters=256, filter_size=(3,3), nonlinearity=tanh, W=GlorotUniform(gain="relu"))
    #l_drop5 = DropoutLayer(l_conv5, p=p)
    #l_conv6 = Conv2DLayer(
    #    l_conv5, num_filters=256, filter_size=(3,3), nonlinearity=tanh, W=GlorotUniform(gain="relu"))
    #l_drop6 = DropoutLayer(l_conv6, p=p)
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

def prepare(args):

    X = T.tensor4('X')

    l_out = get_net(args)
    net_out = get_output(l_out, X, deterministic=True)

    loss = squared_error(net_out, X).mean()
    params = get_all_params(l_out, trainable=True)

    num_epochs, batch_size, learning_rate, momentum = \
        args["num_epochs"], args["batch_size"], args["learning_rate"], args["momentum"]
    if "rmsprop" in args:
        updates = rmsprop(loss, params, learning_rate=learning_rate)
    else:
        updates = nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)

    train_fn = theano.function([X], loss, updates=updates)
    eval_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out)

    return {
        "train_fn":train_fn,
        "eval_fn":eval_fn,
        "out_fn":out_fn,
        "l_out":l_out
    }

def train(args):
    symbols = prepare(args)
    train_fn, eval_fn, out_fn, l_out = symbols["train_fn"], symbols["eval_fn"], symbols["out_fn"], symbols["l_out"]
    X_train = args["X_train"]
    num_epochs, bs = args["num_epochs"], args["batch_size"]

    train_idxs = [x for x in range(0, X_train.shape[0])]

    best_train_loss = float('inf')
    print "epoch,train_loss,has_train_loss_improved,time"
    for epoch in range(0, num_epochs):
        random.shuffle(train_idxs)
        X_train = X_train[train_idxs]
        t0 = time()
        this_losses = []
        b = 0
        while True:
            if b*bs >= X_train.shape[0]:
                break
            X_batch = X_train[b*bs : (b+1)*bs]
            this_loss = train_fn(X_batch)
            this_losses.append(this_loss)
            b += 1
        avg_train_loss = np.mean(this_losses)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_train_loss_ind = 1
            best_model = lasagne.layers.get_all_param_values(l_out)
            with open(args["out_pkl"], "wb") as f:
                pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)
        else:
            best_train_loss_ind = 0
        print "%i,%f,%i,%f" % (epoch+1, avg_train_loss, best_train_loss_ind, time()-t0)
