
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

    x_filename = os.environ["DATA_DIR"] + "/train/train_data.npy"
    X_train = np.load(x_filename).astype("float32")

    #X_train_fake = np.random.normal(0, 1, size=(10, 1, 256, 256))
    #X_train_fake = X_train_fake.astype("float32")

    args = dict()
    args["X_train"] = X_train
    args["num_epochs"] = 10
    args["learning_rate"] = 0.01
    args["batch_size"] = 128
    args["momentum"] = 0.9
    args["p"] = 0

    model = train_ae.train(args)

    print "saving model..."
    with open("model.pkl","wb") as f:
	pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
