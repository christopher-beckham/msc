
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
        seed = 3 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01})
        train_baseline(
            cfg,
            num_epochs=250,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            schedule={201: 0.001}
        )
    #resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
    #resume_legacy=True,
    # training set kappa = 0.81867196628 after 200 epochs
        
    # now we need to resume both of them under a lower learning rate
    if "LOW_RES_N2_BASELINE_CROP_RESUME" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.001})
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.200",
            resume_legacy=True
        )
        seed = 2 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.001})
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.2.model.200",
            resume_legacy=True
        )
        
    
    if "LOW_RES_N2_BASELINE_CROP_HYPERSPHERE_S1" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"mode": "hypersphere", "learn_end": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "out_nonlinearity":linear})
        train_baseline(
            cfg,
            num_epochs=250,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_hypersphere.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            schedule={201: 0.001}
        )




        

    # use squared error loss, but since we still use softmax
    # evaluate the x-ent on the validation set
    # 'sq-err + scale'
    # REDO this experiment
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_SIGM_SCALED" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"sigm_scaled"})
        train_baseline(
            cfg,
            num_epochs=16+50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            schedule={16+1: 0.001},
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_learn-end_sigm-s.1.modelv2.184.bak"
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
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_SIGMOUT_RESUME_S1" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.001, "learn_end":"linear", "out_nonlinearity":sigmoid})
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_learn-end_sigm-out.1.model.200",
            resume_legacy=True
        )
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_SIGMOUT_RESUME_S2" in os.environ:
        seed = 2 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.001, "learn_end":"linear", "out_nonlinearity":sigmoid})
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_learn-end_sigm-out.2.modelv2.200"
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
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_klo_vxent.2.modelv2.50.bak4",
            skip_train=True,
            save_valid_dists="valid_dist/low_res_n2_baseline_crop_klo_vxent.%i" % seed
        )
        # first resume = models/low_res_n2_baseline_crop_klo_vxent.2.modelv2.60.bak (keep at lr = 0.1)
        # second resume = models/low_res_n2_baseline_crop_klo_vxent.2.modelv2.58.bak2 (make lr = 0.01)
        # third resume = /data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_klo_vxent.2.modelv2.82.bak3
        # last model = /data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_klo_vxent.2.modelv2.50.bak4
        
        # ------------
        # stop at 60, then resume but keep at alpha=0.1
        # then 58 epochs after, stop then resume at alpha=0.01


    if "LOW_RES_N2_BASELINE_CROP_KLO_VALID_XENT_S3" in os.environ:
        seed = 3
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":0.1 })
        train_baseline(
            cfg,
            num_epochs=100,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )


        
    if "LOW_RES_N2_BASELINE_CROP_QWK_S1" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=31+50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwk.1.modelv2.38.bak2",
            schedule={31+1: 0.001}
        )
        # earlier = low_res_n2_baseline_crop_qwk.1.modelv2.131.bak


    if "LOW_RES_N2_BASELINE_CROP_QWKREFORM_LEARNEND_S1" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "mode":"qwk_reform", "learn_end":True, "learn_end_options":"normal", "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=31+50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwkreform_learnend.%i" % seed,
            augment=True,
            crop=224
        )

        

        
    if "LOW_RES_N2_BASELINE_CROP_QWK_S1_RESUME" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwk.1.modelv2.131.bak"
        )
    if "LOW_RES_N2_BASELINE_CROP_QWK_S2_RESUME" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwk.2.modelv2.130.bak"
        )


    if "LOW_RES_N2_BASELINE_CROP_QWKCF_BALANCED_BS512_S1" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 512, "l2": 1e-4, "N":2, "qwk_cf":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwkcf_balanced_bs512.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            balanced_minibatches=True,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwkcf_balanced_bs512.1.modelv2.591.bak2",
            debug=True
        )
    # /data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwkcf_balanced_bs512.1.modelv2.56.bak at lr=0.01

    if "LOW_RES_N2_BASELINE_CROP_QWKCF_BALANCED_BS512_S1" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 512, "l2": 1e-4, "N":2, "qwk_cf":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwkcf_balanced_bs512.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            balanced_minibatches=True
        )

    
    if "LOW_RES_N2_BASELINE_CROP_QWK_BALANCED_BS512_S1_LR005" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 512, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk_balanced_bs512_lr0.05.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            balanced_minibatches=True,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwk_balanced_bs512_lr0.05.1.modelv2.166.bak2"
        )
        # resume = /data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwk_balanced_bs512_lr0.05.1.modelv2.250.bak at lr=0.01
    # the latest for this experiment: kappa on training set is 0.883019063441

    if "LOW_RES_N2_BASELINE_CROP_QWK_BS512_S1_LR0001" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 512, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk_bs512_lr0.001.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )
        # resume = /data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_qwk_balanced_bs512_lr0.05.1.modelv2.250.bak at lr=0.01

    if "LOW_RES_N2_BASELINE_CROP_QWK_HYBRID001" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk_hybrid":0.01, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk_hybrid0.01.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )
        
    if "LOW_RES_N2_BASELINE_CROP_QWK_HYBRID01" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk_hybrid":0.1, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk_hybrid0.1.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )

    if "LOW_RES_N2_BASELINE_CROP_QWK_HYBRID1" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk_hybrid":1, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_qwk_hybrid1.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224
        )




    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWKNORMW" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk_normw":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=100,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwknormw.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            schedule={51: 0.001},
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
            resume_legacy=True
        )



        
        
    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWK" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwk.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_resume_with_qwk.1.modelv2.50.bak",
        )
        #resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
        #resume_legacy=True,

    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWK_S2" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=100,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwk.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.2.model.150.bak",
            resume_legacy=True,
            schedule={51: 0.001}
        )
        #resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
        #resume_legacy=True,



        
    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWK_NM" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk_numerator_mean":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwk_nm.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
            resume_legacy=True
        )

        
    if "LOW_RES_N2_BASELINE_CROP_RESUME_KAPPA" in os.environ:
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "kappa":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_kappa.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
            resume_legacy=True
        )

        

    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWK_BS512" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 512, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=100,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwk_bs512.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
            resume_legacy=True
        )
        
    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWK_BALANCED_BS512" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=100,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwk_balanced_bs512.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_resume_with_qwk_bs512.1.modelv2.100.bak",
            skip_train=True,
            save_valid_dists="valid_dist/low_res_n2_baseline_crop_resume_with_qwk_balanced_bs512.%i" % seed
        )
    # resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak"

    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWKCF_BS512" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "qwk_cf":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwkcf_bs512.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_resume_with_qwkcf_bs512.1.modelv2.102.bak2",
            balanced_minibatches=False,
            skip_train=True,
            save_valid_dists="valid_dist/low_res_n2_baseline_crop_resume_with_qwkcf_bs512.%i" % seed
        )
        # first resume
        # resume_legacy=True
        # resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak"
        # second resume
        # resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_resume_with_qwkcf_bs512.1.modelv2.102.bak2"


        
        
    if "LOW_RES_N2_BASELINE_CROP_RESUME_QWKCF_BS512_BUT_LOWER_LR" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 512, "l2": 1e-4, "N":2, "qwk_cf":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_resume_with_qwkcf_bs512_but_lr0.001.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop.1.model.150.bak",
            resume_legacy=True,
            balanced_minibatches=True
        )        

    


    # bug: batch size was 125
    if "LOW_RES_N2_BASELINE_CROP_LOGQWK_BALANCED" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "log_qwk":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_logqwk_balanced.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            balanced_minibatches=True,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_logqwk_balanced.1.modelv2.240.bak"
        )

    if "LOW_RES_N2_BASELINE_CROP_LOGQWK_BALANCED_BS512" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 512, "l2": 1e-4, "N":2, "log_qwk":True, "learning_rate":0.01 })
        train_baseline(
            cfg,
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_logqwk_bs512_balanced.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            balanced_minibatches=True
        )

        



        
    # resume both klo experiments
    if "LOW_RES_N2_BASELINE_CROP_KLO_VALID_XENT_RESUME" in os.environ:
        """
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_klo_vxent.1.model.200",
            resume_legacy=True
        )
        """
        seed = 2
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, { "batch_size": 128, "l2": 1e-4, "N":2, "klo":True, "learning_rate":0.001 })
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_klo_vxent.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_klo_vxent.2.modelv2.82.bak3"
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
            num_epochs=76+50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_relu-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_learn-end_relu-out.1.modelv2.124",
            schedule={77:0.001}
        )
        #124+76 epochs will get us to 200 epochs
        #then at the start of the 201st epochs ('77') we switch to a=0.001 
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_RELUOUT_S2" in os.environ:
        seed = 2 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"linear", "out_nonlinearity":rectify})
        train_baseline(
            cfg,
            num_epochs=250,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_relu-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            schedule={201:0.001}
        )
    if "LOW_RES_N2_BASELINE_CROP_LEARN_END_RELUOUT_S3" in os.environ:
        seed = 3 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.01, "learn_end":"linear", "out_nonlinearity":rectify})
        train_baseline(
            cfg,
            num_epochs=250,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_learn-end_relu-out.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            schedule={201:0.001}
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
    if "LOW_RES_N2_BASELINE_CROP_HACKY_ORDINAL_RESUME" in os.environ:
        seed = 1 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline_hacky_ordinal, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.001, "hacky_ordinal":True})
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_hacky_ordinal.1.model.200"
        )
        seed = 2 # TAKE NOTICE!!!
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net_baseline(resnet_net_224_baseline_hacky_ordinal, {"kappa_loss": False, "batch_size": 128, "l2": 1e-4, "N":2, "learning_rate":0.001, "hacky_ordinal":True})
        train_baseline(
            cfg,
            num_epochs=50,
            data=(X_train, y_train, X_valid, y_valid),
            out_file="output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.%i" % seed,
            augment=True,
            zmuv=True,
            crop=224,
            resume="/data/lisatmp4/beckhamc/models_neat/low_res_n2_baseline_crop_hacky_ordinal.2.modelv2.200"
        )







    


    
