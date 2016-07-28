
import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from lasagne.regularization import *
import sys
sys.path.append("../../modules/")
import helper as hp
import matplotlib.pyplot as plt
#import draw_net
import numpy as np
from skimage import io, img_as_float
import cPickle as pickle
import os
from time import time
from keras.preprocessing.image import ImageDataGenerator
from quadrant_network_dr import *
import glob

# ----------------


def get_test_data(dirname):
    X_left = []
    X_right = []
    tmp = {}
    for filename in glob.glob("%s/*.jpeg" % dirname):
        basename = os.path.basename(filename).replace(".jpeg","")
        #print "basename", basename
        keyname = basename.replace("_left","").replace("_right","")
        #print "keyname", keyname
        if keyname not in tmp:
            tmp[keyname] = {"left":None, "right":None}

        if "left" in basename:
            tmp[keyname]["left"] = basename
        elif "right" in basename:
            tmp[keyname]["right"] = basename
        else:
            raise Exception("left/right error")
    X_left=[]
    X_right=[]
    for key in tmp:
        assert tmp[key]["left"] != None or tmp[key]["right"] != None
        X_left.append(tmp[key]["left"])
        X_right.append(tmp[key]["right"])
    return X_left, X_right


def pred1():

    test_dir = "/data/lisatmp4/beckhamc/test-trim-ben-r400-512"
    cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True, "end_dropout":0.5 })
    with open("models/new_only_augment_b64_zmuv_l2_keras-aug_absorb_fsm_d0.5.1.model.43.bak.bestkappa") as f:
        set_all_param_values(cfg["l_out_pseudo"], pickle.load(f))
    preds_fn = cfg["preds_fn"]
    dist_fn = cfg["dist_fn"]
    X_test_left, X_test_right = get_test_data(test_dir)
    for i in range(0, len(X_test_left)):
        left_image_name = X_test_left[i]
        right_image_name = X_test_right[i]

        this_left_image = hp.load_image_fast("%s/%s.jpeg" % (test_dir,left_image_name), augment=False, zmuv=True)
        this_left_image = np.asarray([this_left_image], dtype="float32")

        this_right_image = hp.load_image_fast("%s/%s.jpeg" % (test_dir,right_image_name), augment=False, zmuv=True)
        this_right_image = np.asarray([this_right_image], dtype="float32")

        sys.stderr.write("left img shape: %s\n" % str(this_left_image.shape))
        sys.stderr.write("right img shape: %s\n" % str(this_right_image.shape))

        left_dist, right_dist = dist_fn(this_left_image, this_right_image)
        for dist in left_dist:
            if sum(np.isnan(dist)) > 0:
                #raise Exception(left_image_name)
                left_pred = "nan"
            else:
                # should only happen once
                left_pred = int(np.round(np.dot(dist, np.arange(0,5))))
        for dist in right_dist:
            if sum(np.isnan(dist)) > 0:
                ###raise Exception(right_image_name)
                right_pred = "nan"
            else:
                # should only happen once
                right_pred = int(np.round(np.dot(dist, np.arange(0,5))))

        print left_image_name + "," + str(left_pred)
        print right_image_name + "," + str(right_pred)



def test_iterator(X_test_left, X_test_right, bs=32):
    b = 0
    while True:
        if b*bs >= len(X_test_left):
            break
        X_left_batch = X_test_left[b*bs:(b+1)*bs]
        X_right_batch = X_test_right[b*bs:(b+1)*bs]
        yield X_left_batch, X_right_batch
        b += 1

        
def pred2():

    test_dir = "/data/lisatmp4/beckhamc/test-trim-ben-r400-512"
    cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True, "end_dropout":0.5 })
    with open("models/new_only_augment_b64_zmuv_l2_keras-aug_absorb_fsm_d0.5.1.model.43.bak.bestkappa") as f:
        set_all_param_values(cfg["l_out_pseudo"], pickle.load(f))
    preds_fn = cfg["preds_fn"]
    dist_fn = cfg["dist_fn_nondet"]
    X_test_left, X_test_right = get_test_data(test_dir)

    num_repeats = 40
    bs = 128
    for epoch in range(0, num_repeats):
        for X_left_batch, X_right_batch in test_iterator(X_test_left, X_test_right, bs):
            left_image_batch = [ hp.load_image_fast("%s/%s.jpeg" % (test_dir,left_image_name), augment=True, zmuv=True) for left_image_name in X_left_batch ]
            right_image_batch = [ hp.load_image_fast("%s/%s.jpeg" % (test_dir,right_image_name), augment=True, zmuv=True) for right_image_name in X_right_batch ]
            left_image_batch = np.asarray(left_image_batch, dtype="float32")
            right_image_batch = np.asarray(right_image_batch, dtype="float32")
            left_dist, right_dist = dist_fn(left_image_batch, right_image_batch)
            # left dist = (bs, 5)
            # right dist = (bs, 5)
            for name, dist in zip(X_left_batch, left_dist):
                if sum(np.isnan(dist)) > 0:
                    pred = "nan"
                else:
                    pred = int(np.round(np.dot(dist, np.arange(0,5))))
                print name + "," + str(pred)
            for name, dist in zip(X_right_batch, right_dist):
                if sum(np.isnan(dist)) > 0:
                    pred = "nan"
                else:
                    pred = int(np.round(np.dot(dist, np.arange(0,5))))
                print name + "," + str(pred)
        

if __name__ == "__main__":
    pred2()
