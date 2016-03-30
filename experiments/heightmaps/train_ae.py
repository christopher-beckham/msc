import imp
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

def prepare(args):

    X = T.tensor4('X')

    config = imp.load_source("config", args["config"])
    l_out = config.get_net(args)
    net_out = get_output(l_out, X, deterministic=True)
    if "in_pkl" in args:
        with open(args["in_pkl"]) as f:
            set_all_param_values(l_out, pickle.load(f))
            sys.stderr.write("loading existing model at %s\n" % args["in_pkl"])

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
