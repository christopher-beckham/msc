import sys
import gzip
import cPickle as pickle
import numpy as np

def label_to_one_hot(x, num_classes):
    """
    convert a label to one-hot
    :param x:
    :param num_classes:
    :return:
    """
    this_onehot = []
    for elem in x:
        one_hot_vector = [0] * num_classes
        one_hot_vector[elem] = 1
        this_onehot.append(one_hot_vector)
    this_onehot = np.asarray(this_onehot, dtype="float32")
    return this_onehot

def load_mnist(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename) as f:
        dat = pickle.load(f)
    train_set, valid_set, test_set = dat
    Xt, yt = train_set
    Xv, yv = valid_set
    Xtest, ytest = test_set
    Xt = Xt.reshape( Xt.shape[0], 1, 28, 28 )
    Xv = Xv.reshape( Xv.shape[0], 1, 28, 28 )
    Xtest = Xtest.reshape( Xtest.shape[0], 1, 28, 28 )

    yt = label_to_one_hot(yt, num_classes=10)
    yv = label_to_one_hot(yv, num_classes=10)
    ytest = label_to_one_hot(ytest, num_classes=10)

    # TODO: don't do test

    return Xt, yt, Xv, yv

if __name__ == '__main__':
    #pass
    load_mnist("../../../data/mnist.pkl.gz")