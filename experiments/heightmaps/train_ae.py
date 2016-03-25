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

def get_net(args):
    pass

def prepare(args):

    X = T.tensor4('X')

    l_out = get_net(args)
    for layer in get_all_layers(l_out):
        sys.stderr.write( "%s, %s\n" % (layer, layer.output_shape) )
    sys.stderr.write("number of params: %i\n" % count_params(l_out))

    net_out = get_output(l_out, X)

    loss = squared_error(net_out, X).mean()
    params = get_all_params(l_out, trainable=True)

    num_epochs, batch_size, learning_rate, momentum = \
        args["num_epochs"], args["batch_size"], args["learning_rate"], args["momentum"]
    updates = nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)

    train_fn = theano.function([X, y], loss, updates=updates)
    eval_fn = theano.function([X, y], loss)
    out_fn = theano.function([X], net_out)

    return {
        "train_fn":train_fn,
        "eval_fn":eval_fn,
        "out_fn":out_fn,
        "l_out":l_out
    }

def train(args):
    symbols = prepare(args)
    train_fn, eval_fn, out_fn = symbols["train_fn"], symbols["eval_fn"], symbols["out_fn"]
    X_train = args["X_train"]
    num_epochs, bs = args["num_epochs"], args["batch_size"]

    train_idxs = [x for x in range(0, X_train.shape[0])]

    print "train_loss,time"
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
        print "%f,%f" % (np.mean(this_losses), time()-t0)