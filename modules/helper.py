import cPickle as pickle
import gzip
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from skimage import img_as_float
from nolearn.lasagne import *

"""
mnist
"""

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
    return ( (Xt,yt), (Xv, yv), (Xtest, ytest) )

def load_cluttered_mnist_train_only(filename="../data/mnist_cluttered_60x60_6distortions.npz"):
    dat = np.load(filename)
    Xt = dat["x_train"]
    Xt = Xt.reshape( (Xt.shape[0], 1, 60, 60) )
    yt = dat["y_train"]
    yt = np.argmax(yt, axis=-1)
    return Xt, yt

"""
nolearn
"""

def save_stats_at_every(schedule, filename):
    def after_epoch(net, info):
        if info[-1]["epoch"] % schedule == 0:
            write_stats(info, filename)
    return after_epoch

def write_stats(info, filename):
    f = open("%s.csv" % filename, "wb")
    # [valid_accuracy',  'dur', 'abs_error']
    f.write("epoch,train_loss,train_loss_best,valid_loss,valid_loss_best,valid_accuracy,dur\n")
    for row in info:
            f.write("%f,%f,%f,%f,%f,%f,%f\n" % (row["epoch"], row["train_loss"], \
                    row["train_loss_best"], row["valid_loss"], row["valid_loss_best"], row["valid_accuracy"], row["dur"]))
    #pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def save_stats_on_best(filename):
    def after_epoch(net, info):
        if info[-1]["valid_loss_best"]:
            write_stats(info, filename)
    return after_epoch

def save_model_on_best(filename):
    def after_epoch(net, info):
        if info[-1]["valid_loss_best"]:
            net.save_weights_to(filename)
    return after_epoch

def shuffle(*arrays):
    # https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]

# TODO: no longer needed
class ShufflingBatchIterator(BatchIterator):
    # https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShufflingBatchIterator, self).__iter__():
            yield res

class ImageBatchIterator(BatchIterator):
    def __init__(self, filenames, prefix, zmuv=False, *args, **kwds):
        super(ImageBatchIterator, self).__init__(*args, **kwds)
        self.filenames = filenames
        self.prefix = prefix
        self.zmuv = zmuv
    def transform(self, Xb, yb):
        filenames = np.asarray( [ self.filenames[int(x)] for x in Xb.flatten().tolist() ] )
        if self.prefix == "":
            Xb_actual = np.asarray(
                [ load_image(x, zmuv=self.zmuv) for x in filenames ], dtype="float32" 
            )
        else:
            Xb_actual = np.asarray(
                [ load_image(self.prefix + os.path.sep + x, zmuv=self.zmuv) for x in filenames ], dtype="float32" 
            )
        return Xb_actual, yb

"""
diabetic retinopathy
"""

def weighted_kappa(human_rater, actual_rater, num_classes=5):
    assert len(human_rater) == len(actual_rater)
    def sum_matrix(X, Y):
        assert len(X) == len(Y)
        assert len(X[0]) == len(Y[0])
        sum = 0
        for i in range(0, len(X)):
            for j in range(0, len(X[0])):
                sum += X[i][j]*Y[i][j]
        return sum
    # compute W
    W = [ [float(0) for y in range(0, num_classes)] for x in range(0, num_classes) ]
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            num = (i-j)**2
            den = (float(num_classes) - 1)**2
            W[i][j] = num # / den
    # compute O
    O = [ [float(0) for y in range(0, num_classes)] for x in range(0, num_classes) ]
    # rows = human_rater
    # cols = actual_rater
    for i in range(0, len(actual_rater)):
        O[ human_rater[i] ][ actual_rater[i] ] += 1
    # normalise O
    total = sum([sum(x) for x in O])
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            O[i][j] = O[i][j] / total
    # compute E
    total = sum([sum(x) for x in O])
    E = [ [float(0) for y in range(0, num_classes)] for x in range(0, num_classes) ]
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            # E_ij = row(i) total * col(j) total / total
            col_j = [ O[x][j] for x in range(0, len(O[0])) ]
            E[i][j] = sum(O[i]) * sum(col_j) / total
    # compute kappa
    kappa = 1 - (sum_matrix(W, O) / sum_matrix(W, E))
    return kappa


def load_image(filename, rotate=False, zmuv=False):
    img = io.imread(filename)
    img = img_as_float(img)
    if rotate:
        img = np.rot90(img, random.randint(0, 3))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.asarray( [ img[...,0], img[...,1], img[...,2] ] )
    else:
        img = np.asarray( [ img ] )
    if zmuv:
        for i in range(0, img.shape[0]):
            img[i, ...] = (img[i, ...] - np.mean(img[i, ...])) / np.std(img[i,...])
    return img

def load_dr_train_labels(filename):
    tps = []
    with open(filename) as f:
        f.readline()
        for line in f:
            line = line.rstrip().split(",")
            tps.append( (line[0] + ".jpeg", line[1]) )
    random.shuffle(tps)
    return map(list, zip(*tps))

def get_kappa_loss(num_classes):
    def this_kappa_fn(pred_vector, actual_vector):
        return (T.dot(pred_vector, T.arange(0,num_classes)) - actual_vector)**2
    return this_kappa_fn
