
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

# -----------------------------

if __name__ == "__main__":

    with open("dr.pkl") as f:
        dat = pickle.load(f)
    X_left, X_right, y_left, y_right = dat
    X_left = np.asarray(X_left)
    X_right = np.asarray(X_right)
    y_left = np.asarray(y_left, dtype="int32")
    y_right = np.asarray(y_right, dtype="int32")
    np.random.seed(0)
    idxs = [x for x in range(0, len(X_left))]
    np.random.shuffle(idxs)

    X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
    X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
    y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
    y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
    X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
    X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
    y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
    y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
    # ok, fix now
    X_train = np.hstack((X_train_left, X_train_right))
    X_valid = np.hstack((X_valid_left,X_valid_right))
    y_train = np.hstack((y_train_left,y_train_right))
    y_valid = np.hstack((y_valid_left,y_valid_right))

    # baseline experiment
    # 'x-ent'
    if "LOW_RES_N2_BASELINE_CROP" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    # use squared error loss, but since we still use softmax
    # evaluate the x-ent on the validation set
    # 'sq-err'
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"linear"})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end.deleteme.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    # use squared error loss, but since we still use softmax
    # evaluate the x-ent on the validation set
    # 'sq-err + scale'
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_SIGM_SCALED" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"sigm_scaled"})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    # use the squared error loss, but this time use sigmoid nonlinearity for 'h'
    # this incorrectly has x-ent applied to the validation set (fix)
    # 'sq-err + sigm'
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_SIGMOUT" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"linear", "out_nonlinearity":sigmoid})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

        
    # use the squared error loss, but this time use sigmoid nonlinearity for 'h'
    # this incorrectly has x-ent applied to the validation set (fix)
    # 'sq-err + sigm + scale'
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_SIGM_SCALED_SIGMOUT" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"sigm_scaled", "out_nonlinearity":sigmoid})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s_sigm-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    # --------------

    if "LOW_RES_N2_BASELINE_CROP_KLO_VALID_XENT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":0.1 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent.%i" % seed,
            augment=True,
            zmuv=True,
            schedule={61:0.01},
            crop=224
        )

    # seed 2 redo
    if "LOW_RES_N2_BASELINE_CROP_KLO_VALID_XENT_S2" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="models/low_res_n2_baseline_crop_klo_vxent.2.modelv2.58.bak2"
        )
        # first resume = models/low_res_n2_baseline_crop_klo_vxent.2.modelv2.60.bak (keep at lr = 0.1)
        # second resume = models/low_res_n2_baseline_crop_klo_vxent.2.modelv2.58.bak2 (make lr = 0.01)
        # ------------
        # stop at 60, then resume but keep at alpha=0.1
        # then 58 epochs after, stop then resume at alpha=0.01


    if "LOW_RES_N2_BASELINE_CROP_KLO_VALID_XENT_S1_RAPID" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":1. })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent_rapid.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    if "LOW_RES_N2_BASELINE_CROP_KLO_VALID_XENT_ENTRREG" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":0.1, "entr_regulariser":True })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent_entrreg.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )
    
    if "LOW_RES_N2_BASELINE_CROP_KLO_VALID_XENT_ENTRREG_S2" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":0.1, "entr_regulariser":True })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent_entrreg.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )


            
    # use the kappa exp loss after training for 50 epochs on x-ent
    if "LOW_RES_N2_BASELINE_CROP_KLO_FINE_TUNE" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "klo":True})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_klo.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="models/low_res_n2_baseline_crop_resume_klo.1.model.7.bak",
            resume_legacy=True
        )
    #models/low_res_n2_baseline_crop.1.model.50.bak

    # use the kappa exp loss after training for 50 epochs on x-ent
    if "LOW_RES_N2_BASELINE_CROP_KLO_FINE_TUNE_EARLIER" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "klo":True})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_klo_at25epochs.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="models/low_res_n2_baseline_crop.1.model.25.bak",
            resume_legacy=True
        )
    #models/low_res_n2_baseline_crop.1.model.50.bak

        
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_SIGM_SCALED_RELUOUT" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"sigm_scaled", "out_nonlinearity":rectify})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s_relu-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_RELUOUT" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"linear", "out_nonlinearity":rectify})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_relu-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )
    



    
    # ordinal hacky baseline
    #
    if "LOW_RES_N2_BASELINE_CROP_HACKY_ORDINAL" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline_hacky_ordinal, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "hacky_ordinal":True})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    if "LOW_RES_N2_BASELINE_CROP_HACKY_ORDINAL_S2" in os.environ:
        seed = 2 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline_hacky_ordinal, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "hacky_ordinal":True})
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )


    
    # ---------------------

    # hybrid loss experiments
    # these use crops

    if "LOW_RES_N2_BASELINE_CROP_KL01" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "kl":1e-1, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_kl0.1.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224)
#            resume="models/low_res_n2_baseline_crop_kl0.1.1.modelv2.171.bak"
#        )

    if "LOW_RES_N2_BASELINE_CROP_KL05" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "kl":0.5, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_kl0.5.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    if "LOW_RES_N2_BASELINE_CROP_KL10" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "kl":1.0, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_kl1.0.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224)
#            resume="models/low_res_n2_baseline_crop_kl1.0.1.modelv2.127.bak"
#        )


    # -----------------

    if "LOW_RES_N2_BASELINE_CROP_KL20" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "kl":2.0, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_kl2.0.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    if "LOW_RES_N2_BASELINE_CROP_KL30" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "kl":3.0, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_kl3.0.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )
