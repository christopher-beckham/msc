import cPickle as pickle
import gzip
from matplotlib import pyplot as plt
import numpy as np

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

def save_stats_at_every(schedule, filename):
    def after_epoch(net, info):
        if info[-1]["epoch"] % schedule == 0:
            write_stats(info, filename)
    return after_epoch

def write_stats(info, filename):
    f = open("%s.csv" % filename, "wb")
    # [valid_accuracy',  'dur', 'abs_error']
    f.write("epoch,train_loss,train_loss_best,valid_loss,valid_loss_best,dur,abs_error\n")
    for row in info:
        if "abs_error" in row:
            f.write("%f,%f,%f,%f,%f,%f,%f\n" % (row["epoch"], row["train_loss"], \
                row["train_loss_best"], row["valid_loss"], row["valid_loss_best"], row["dur"], row["abs_error"]))
        else:
            f.write("%f,%f,%f,%f,%f,%f\n" % (row["epoch"], row["train_loss"], \
                row["train_loss_best"], row["valid_loss"], row["valid_loss_best"], row["dur"]))
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
