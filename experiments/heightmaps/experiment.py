
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
import train_ae

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

   # X_all, y_all = np.load(x_filename), np.load(y_filename)

    X_train_fake = np.random.normal(0, 1, size=(300, 1, 256, 256))

    args = dict()
    args["X_train"] = X_train_fake
    args["num_epochs"] = 200
    args["learning_rate"] = 0.01
    args["batch_size"] = 128
    args["momentum"] = 0.9
    args["p"] = 0

    train_ae.train(args)