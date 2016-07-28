
# coding: utf-8

# In[1]:

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


##
# In[3]:

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

X_train_left = X_left[idxs][0 : int(0.9*X_left.shape[0])]
X_train_right = X_right[idxs][0 : int(0.9*X_right.shape[0])]
y_train_left = y_left[idxs][0 : int(0.9*y_left.shape[0])]
y_train_right = y_right[idxs][0 : int(0.9*y_left.shape[0])]

X_valid_left = X_left[idxs][int(0.9*X_left.shape[0]) ::]
X_valid_right = X_right[idxs][int(0.9*X_right.shape[0]) ::]
y_valid_left = y_left[idxs][int(0.9*y_left.shape[0]) ::]
y_valid_right = y_right[idxs][int(0.9*y_right.shape[0]) ::]

# -----

# In[9]:

def get_net(net_fn, args={}):
    # ----
    X_left = T.tensor4('X_left')
    X_right = T.tensor4('X_right')
    y_left = T.ivector('y_left')
    y_right = T.ivector('y_right')
    # ----
    
    cfg = net_fn(args)
    l_out_left = cfg["l_out_left"]
    l_out_right = cfg["l_out_right"]
    l_in_left = cfg["l_in_left"]
    l_in_right = cfg["l_in_right"]
    l_out_pseudo = cfg["l_out_pseudo"]
                        
    net_out_left, net_out_right = get_output(
        [l_out_left, l_out_right],
        {l_in_left: X_left, l_in_right: X_right}
    )
    net_out_left_det, net_out_right_det = get_output(
        [l_out_left, l_out_right],
        {l_in_left: X_left, l_in_right: X_right},
        deterministic=True
    )
    
    if not args["kappa_loss"]:
        loss = categorical_crossentropy(net_out_left, y_left).mean() + categorical_crossentropy(net_out_right, y_right).mean()
        loss_det = categorical_crossentropy(net_out_left_det, y_left).mean() + categorical_crossentropy(net_out_right_det, y_right).mean()
    else:
        if "hybrid_loss" not in args:
            loss = hp.get_kappa_loss(5)(net_out_left, y_left).mean() + hp.get_kappa_loss(5)(net_out_right, y_right).mean()
            loss_det = hp.get_kappa_loss(5)(net_out_left_det, y_left).mean() + hp.get_kappa_loss(5)(net_out_right_det, y_right).mean()
        else:
            hl = args["hybrid_loss"]
            sys.stderr.write("hybrid loss factor: %f\n" % hl)
            loss = categorical_crossentropy(net_out_left, y_left).mean() + categorical_crossentropy(net_out_right, y_right).mean() + \
                   hl*hp.get_kappa_loss(5)(net_out_left, y_left).mean() + hl*hp.get_kappa_loss(5)(net_out_right, y_right).mean()
            loss_det = categorical_crossentropy(net_out_left_det, y_left).mean() + categorical_crossentropy(net_out_right_det, y_right).mean() + \
                       hl*hp.get_kappa_loss(5)(net_out_left_det, y_left).mean() + hl*hp.get_kappa_loss(5)(net_out_right_det, y_right).mean()
            
        #raise NotImplementedError()
    
    if "l2" in args:
        sys.stderr.write("adding l2: %f\n" % args["l2"])
        loss += args["l2"]*regularize_network_params(l_out_pseudo, l2)
        loss_det += args["l2"]*regularize_network_params(l_out_pseudo, l2)
    params = get_all_params(l_out_pseudo, trainable=True)
    sys.stderr.write("params: %s\n" % str(params))
    if "max_norm" in args:
        grads = total_norm_constraint( T.grad(loss, params), max_norm=args["max_norm"])
    else:
        grads = T.grad(loss, params)
    learning_rate = theano.shared(floatX(0.01)) if "learning_rate" not in args else theano.shared(floatX(args["learning_rate"]))
    momentum = 0.9 if "momentum" not in args else args["momentum"]
    if "rmsprop" in args:
        sys.stderr.write("using rmsprop instead of nesterov momentum...\n")
        updates = rmsprop(grads, params, learning_rate=learning_rate)
    else:
        updates = nesterov_momentum(grads, params, learning_rate=learning_rate, momentum=momentum)
    # index fns
    bs = args["batch_size"]
    train_fn = theano.function(inputs=[X_left,X_right,y_left,y_right], outputs=loss, updates=updates)
    loss_fn = theano.function(inputs=[X_left,X_right,y_left,y_right], outputs=loss_det)

    preds_fn = theano.function(
        inputs=[X_left,X_right],
        outputs=[ T.argmax(net_out_left_det,axis=1), T.argmax(net_out_right_det,axis=1) ]
    )
    dist_fn = theano.function(
        inputs=[X_left, X_right],
        outputs=[ net_out_left_det, net_out_right_det ]
    )
    dist_fn_nondet = theano.function(
        inputs=[X_left, X_right],
        outputs=[net_out_left, net_out_right]
    )
    
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "preds_fn": preds_fn,
        "dist_fn": dist_fn,
        "dist_fn_nondet": dist_fn_nondet,
        "l_out_left": l_out_left,
        "l_out_right": l_out_right,
        "l_out_pseudo": l_out_pseudo,
        "learning_rate": learning_rate,
        "bs": bs,
        "kappa_loss": args["kappa_loss"]
    }


# In[24]:

def net(args={}):
    
    #conv_p, fc_p = args["fc_p"]
    
    l_in_left = InputLayer( (None, 3, 512, 512) )
    l_in_right = InputLayer( (None, 3, 512, 512) )
    # left image
    l_topleft_left = SliceLayer( SliceLayer(l_in_left, indices=slice(0,256), axis=2), indices=slice(0,256), axis=3 )
    l_bottomleft_left = SliceLayer( SliceLayer(l_in_left, indices=slice(256,512), axis=2), indices=slice(0,256), axis=3 )
    l_topright_left = SliceLayer( SliceLayer(l_in_left, indices=slice(0,256), axis=2), indices=slice(256,512), axis=3 )
    l_bottomright_left = SliceLayer( SliceLayer(l_in_left, indices=slice(256,512), axis=2), indices=slice(256,512), axis=3 )
    # right image
    l_topleft_right = SliceLayer( SliceLayer(l_in_right, indices=slice(0,256), axis=2), indices=slice(0,256), axis=3 )
    l_bottomleft_right = SliceLayer( SliceLayer(l_in_right, indices=slice(256,512), axis=2), indices=slice(0,256), axis=3 )
    l_topright_right = SliceLayer( SliceLayer(l_in_right, indices=slice(0,256), axis=2), indices=slice(256,512), axis=3 )
    l_bottomright_right = SliceLayer( SliceLayer(l_in_right, indices=slice(256,512), axis=2), indices=slice(256,512), axis=3 ) 
    
    sys.stderr.write("fc_p = %f\n" % args["fc_p"])
    
    def net_block(quadrant, dd):

        sys.stderr.write( str(dd.keys()) + "\n")
        
        #l_in = InputLayer(
        #    shape=(None, 3, 256, 256),
        #)
        # { "type": "CONV", "num_filters": 32, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" },
        l_conv1 = Conv2DLayer(
            quadrant,
            num_filters=32,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv1" not in dd else dd["l_conv1"].W,
            b=Constant(0.) if "l_conv1" not in dd else dd["l_conv1"].b
        )
        l_pool1 = MaxPool2DLayer(
            l_conv1,
            pool_size=(3,3),
            stride=2
        )
        # { "type": "CONV", "dropout": 0.1, "num_filters": 64, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" }
        #l_dropout1 = DropoutLayer(l_pool1, p=0.1)
        l_conv2 = Conv2DLayer(
            l_pool1,
            num_filters=64,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv2" not in dd else dd["l_conv2"].W,
            b=Constant(0.) if "l_conv2" not in dd else dd["l_conv2"].b
        )
        l_pool2 = MaxPool2DLayer(
            l_conv2,
            pool_size=(3,3),
            stride=2
        )
        # { "type": "CONV", "dropout": 0.1, "num_filters": 128, "filter_size": 3, "nonlinearity": "LReLU" },
        #l_dropout2 = DropoutLayer(l_pool2, p=0.1)
        l_conv3 = Conv2DLayer(
            l_pool2,
            num_filters=128,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv3" not in dd else dd["l_conv3"].W,
            b=Constant(0.) if "l_conv3" not in dd else dd["l_conv3"].b
        )
        # { "type": "CONV", "dropout": 0.1, "num_filters": 128, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" },
        #l_dropout3 = DropoutLayer(l_conv3, p=0.1)
        l_conv4 = Conv2DLayer(
            l_conv3,
            num_filters=128,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv4" not in dd else dd["l_conv4"].W,
            b=Constant(0.) if "l_conv4" not in dd else dd["l_conv4"].b
        )
        l_pool3 = MaxPool2DLayer(
            l_conv4,
            pool_size=(3,3),
            stride=2
        )
        # { "type": "CONV", "dropout": 0.1, "num_filters": 128, "filter_size": 3, "pool_size": 3, "pool_stride": 2, "nonlinearity": "LReLU" },
        #l_dropout4 = DropoutLayer(l_pool3, p=0.1)
        l_conv5 = Conv2DLayer(
            l_pool3,
            num_filters=256,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv5" not in dd else dd["l_conv5"].W,
            b=Constant(0.) if "l_conv5" not in dd else dd["l_conv5"].b
        )
        #l_dropout5 = DropoutLayer(l_conv5, p=0.1)
        l_conv6 = Conv2DLayer(
            l_conv5,
            num_filters=256,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv6" not in dd else dd["l_conv6"].W,
            b=Constant(0.) if "l_conv6" not in dd else dd["l_conv6"].b
        )
        # maxpool size 3 stride 2
        l_pool4 = MaxPool2DLayer(
            l_conv6,
            pool_size=(3,3),
            stride=2
        )
        # { "type": "CONV", "dropout": 0.1, "num_filters": 256, "filter_size": 3, "pool_size": 2, "pool_stride": 2, "nonlinearity": "LReLU" },
        #l_dropout5 = DropoutLayer(l_pool4, p=0.1)
        l_conv7 = Conv2DLayer(
            l_pool4,
            num_filters=512,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv7" not in dd else dd["l_conv7"].W,
            b=Constant(0.) if "l_conv7" not in dd else dd["l_conv7"].b
        )
        #l_dropout6 = DropoutLayer(l_conv6, p=0.1)
        l_conv8 = Conv2DLayer(
            l_conv7,
            num_filters=512,
            filter_size=(3,3),
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_conv8" not in dd else dd["l_conv8"].W,
            b=Constant(0.) if "l_conv8" not in dd else dd["l_conv8"].b
        )
        
        l_pool5 = MaxPool2DLayer(
            l_conv8,
            pool_size=(2,2),
            stride=2
        )

        # { "type": "FC", "dropout": 0.5, "num_units": 2048, "pool_size": 2, "nonlinearity": "LReLU" },
        #l_dropout7 = lasagne.layers.DropoutLayer(l_conv7, p=0.5)
        l_hidden1 = lasagne.layers.DenseLayer(
            l_pool5,
            num_units=512,
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_hidden1" not in dd else dd["l_hidden1"].W,
            b=Constant(0.) if "l_hidden1" not in dd else dd["l_hidden1"].b
        )

        #l_pool6 = lasagne.layers.FeaturePoolLayer(
        #    l_hidden1,
        #    pool_size=2
        #)
        
        # { "type": "FC", "dropout": 0.5, "num_units": 2048, "pool_size": 2, "nonlinearity": "LReLU" },
        #l_dropout8 = lasagne.layers.DropoutLayer(l_hidden1, p=args["fc_p"])
        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=512,
            nonlinearity=leaky_rectify,
            W=GlorotUniform(gain="relu") if "l_hidden2" not in dd else dd["l_hidden2"].W,
            b=Constant(0.) if "l_hidden2" not in dd else dd["l_hidden2"].b
        )
        
        #l_pool7 = lasagne.layers.FeaturePoolLayer(
        #    l_hidden2,
        #    pool_size=2
        #)
        
        # { "type": "OUTPUT", "dropout": 0.5, "nonlinearity": "sigmoid" }
        #l_dropout8 = lasagne.layers.DropoutLayer(l_pool7, p=0.5)
        #l_out = lasagne.layers.DenseLayer(
        #    l_dropout8,
        #    num_units=5,
        #    nonlinearity=softmax,
        #    W=GlorotUniform()
        #)
        #return l_out

        #l_out = l_pool7
        #l_out = l_hidden2

        return {
            "l_conv1": l_conv1,
            "l_conv2": l_conv2,
            "l_conv3": l_conv3,
            "l_conv4": l_conv4,
            "l_conv5": l_conv5,
            "l_conv6": l_conv6,
            "l_conv7": l_conv7,
            "l_conv8": l_conv8,
            "l_hidden1": l_hidden1,
            "l_hidden2": l_hidden2,
            "l_out": l_hidden2
        }

    topleft_conv_left = net_block(l_topleft_left, {})
    bottomleft_conv_left = net_block(l_bottomleft_left, topleft_conv_left)
    topright_conv_left = net_block(l_topright_left, topleft_conv_left)
    bottomright_conv_left = net_block(l_bottomright_left, topleft_conv_left)

    topleft_conv_right = net_block(l_topleft_right, topleft_conv_left)
    bottomleft_conv_right = net_block(l_bottomleft_right, topleft_conv_left)
    topright_conv_right = net_block(l_topright_right, topleft_conv_left)
    bottomright_conv_right = net_block(l_bottomright_right, topleft_conv_left)

    for layer in get_all_layers(topleft_conv_left["l_out"]):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    l_concat_left = ConcatLayer([
        topleft_conv_left["l_out"], 
        bottomleft_conv_left["l_out"], 
        topright_conv_left["l_out"], 
        bottomright_conv_left["l_out"],   
    ])

    l_concat_right = ConcatLayer([
        topleft_conv_right["l_out"], 
        bottomleft_conv_right["l_out"], 
        topright_conv_right["l_out"], 
        bottomright_conv_right["l_out"],   
    ])

    l_merge = ElemwiseSumLayer(
        [l_concat_left, l_concat_right]
    )
    
    l_dropout = DropoutLayer(l_merge, p=args["fc_p"])

    l_out = lasagne.layers.DenseLayer(
        l_dropout,
        num_units=5,
        nonlinearity=softmax,
        W=GlorotUniform()
    )

    sys.stderr.write("number of params: %i\n" % count_params(l_out))
    
    return {"l_out": l_out, "l_in_left": l_in_left, "l_in_right": l_in_right}


def residual_block(layer, n_out_channels, prefix, stride=1, dd={}, nonlinearity=rectify):
    conv = layer
    if stride > 1:
        layer = Pool2DLayer(layer, pool_size=1, stride=stride, mode="average_inc_pad")
    if (n_out_channels != layer.output_shape[1]):
        diff = n_out_channels-layer.output_shape[1]
        if diff % 2 == 0: 
            width_tp = ((diff/2, diff/2),)
        else:
            width_tp = (((diff/2)+1, diff/2),)
        layer = pad(layer, batch_ndim=1, width=width_tp)
    conv = Conv2DLayer(conv, 
                       num_filters=n_out_channels,
                       filter_size=(3,3), 
                       stride=(stride,stride), 
                       pad=(1,1), 
                       nonlinearity=linear, 
                       W=HeNormal(gain="relu") if prefix+"_1" not in dd else dd[prefix+"_1"].W,
                       b=Constant(0.) if prefix+"_1" not in dd else dd[prefix+"_1"].b)
    if prefix+"_1" not in dd:
        dd[prefix+"_1"] = conv
        print prefix+"_1"
    conv = BatchNormLayer(conv,
                          beta=Constant(0.) if prefix+"_bn1" not in dd else dd[prefix+"_bn1"].beta, 
                          gamma=Constant(1.) if prefix+"_bn1" not in dd else dd[prefix+"_bn1"].gamma)
    if prefix+"_bn1" not in dd:
        dd[prefix+"_bn1"] = conv
        print prefix+"_bn1"
    conv = NonlinearityLayer(conv, nonlinearity=nonlinearity)
    conv = Conv2DLayer(conv, 
                       num_filters=n_out_channels,
                       filter_size=(3,3), 
                       stride=(1,1), 
                       pad=(1,1), 
                       nonlinearity=linear, 
                       W=HeNormal(gain="relu") if prefix+"_2" not in dd else dd[prefix+"_2"].W,
                       b=Constant(0.) if prefix+"_2" not in dd else dd[prefix+"_2"].b)
    if prefix+"_2" not in dd:
        dd[prefix+"_2"] = conv
        print prefix+"_2"
    conv = BatchNormLayer(conv,
                          beta=Constant(0.) if prefix+"_bn2" not in dd else dd[prefix+"_bn2"].beta, 
                          gamma=Constant(1.) if prefix+"_bn2" not in dd else dd[prefix+"_bn2"].gamma)
    if prefix+"_bn2" not in dd:
        dd[prefix+"_bn2"] = conv
        print prefix+"_bn2"
    return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity=nonlinearity)


def resnet(quadrant, dd, first_time, nf=[32, 64, 128, 256], dropout_p=None):
    # 34-layer resnet as per:
    # https://arxiv.org/pdf/1512.03385v1.pdf
    layer = Conv2DLayer(quadrant, 
                        num_filters=32, 
                        filter_size=7, 
                        stride=2, 
                        nonlinearity=rectify, 
                        pad='same', 
                        W=HeNormal(gain="relu") if "conv1" not in dd else dd["conv1"].W,
                        b=Constant(0.) if "conv1" not in dd else dd["conv1"].b)
    if "conv1" not in dd:
        dd["conv1"] = layer
        print "conv1"
    layer = MaxPool2DLayer(layer, pool_size=3, stride=2)
    for i in range(2):
        layer = residual_block(layer, nf[0], prefix="a%i" % i, dd=dd)
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)
    layer = residual_block(layer, nf[1], prefix="aa", stride=2, dd=dd)
    for i in range(2):
        layer = residual_block(layer, nf[1], prefix="b%i" % i, dd=dd)
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)
    layer = residual_block(layer, nf[2], prefix="bb%", stride=2, dd=dd)
    for i in range(2):
        layer = residual_block(layer, nf[2], prefix="c%i" % i, dd=dd)
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)
    layer = residual_block(layer, nf[3], prefix="cc", stride=2, dd=dd)
    for i in range(2):
        layer = residual_block(layer, nf[3], prefix="dd%i" % i, dd=dd)
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)
    layer = Pool2DLayer(layer, pool_size=8, stride=1, mode="average_inc_pad")
    #layer = DenseLayer(layer, 
    #                   num_units=5, 
    #                   nonlinearity=softmax, 
    #                   W=HeNormal() if "softmax" not in dd else dd["softmax"])
    layer = FlattenLayer(layer)
    
    if first_time:
        return dd, layer
    else:
        return layer


def resnet_net(args={}):
    
    #conv_p, fc_p = args["fc_p"]



    if "dropout_p" in args:
        dropout_p = args["dropout_p"]
        sys.stderr.write("dropout_p: %f\n" % dropout_p)
    else:
        dropout_p = None
    
    l_in_left = InputLayer( (None, 3, 512, 512) )
    l_in_right = InputLayer( (None, 3, 512, 512) )
    # left image
    l_topleft_left = SliceLayer( SliceLayer(l_in_left, indices=slice(0,256), axis=2), indices=slice(0,256), axis=3 )
    l_bottomleft_left = SliceLayer( SliceLayer(l_in_left, indices=slice(256,512), axis=2), indices=slice(0,256), axis=3 )
    l_topright_left = SliceLayer( SliceLayer(l_in_left, indices=slice(0,256), axis=2), indices=slice(256,512), axis=3 )
    l_bottomright_left = SliceLayer( SliceLayer(l_in_left, indices=slice(256,512), axis=2), indices=slice(256,512), axis=3 )
    # right image
    l_topleft_right = SliceLayer( SliceLayer(l_in_right, indices=slice(0,256), axis=2), indices=slice(0,256), axis=3 )
    l_bottomleft_right = SliceLayer( SliceLayer(l_in_right, indices=slice(256,512), axis=2), indices=slice(0,256), axis=3 )
    l_topright_right = SliceLayer( SliceLayer(l_in_right, indices=slice(0,256), axis=2), indices=slice(256,512), axis=3 )
    l_bottomright_right = SliceLayer( SliceLayer(l_in_right, indices=slice(256,512), axis=2), indices=slice(256,512), axis=3 ) 

    dd, topleft_conv_left = resnet(l_topleft_left, {}, True, dropout_p=dropout_p)
    bottomleft_conv_left = resnet(l_bottomleft_left, dd, False, dropout_p=dropout_p)
    topright_conv_left = resnet(l_topright_left, dd, False, dropout_p=dropout_p)
    bottomright_conv_left = resnet(l_bottomright_left, dd, False, dropout_p=dropout_p)

    topleft_conv_right = resnet(l_topleft_right, dd, False, dropout_p=dropout_p)
    bottomleft_conv_right = resnet(l_bottomleft_right, dd, False, dropout_p=dropout_p)
    topright_conv_right = resnet(l_topright_right, dd, False, dropout_p=dropout_p)
    bottomright_conv_right = resnet(l_bottomright_right, dd, False, dropout_p=dropout_p)

    for layer in get_all_layers(topleft_conv_left):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    """
    l_concat_left = ElemwiseSumLayer([
        topleft_conv_left["l_out"], 
        bottomleft_conv_left["l_out"], 
        topright_conv_left["l_out"], 
        bottomright_conv_left["l_out"],   
    ])

    l_concat_right = ElemwiseSumLayer([
        topleft_conv_right["l_out"], 
        bottomleft_conv_right["l_out"], 
        topright_conv_right["l_out"], 
        bottomright_conv_right["l_out"],   
    ])
    """

    l_merge_left = ElemwiseSumLayer([
        topleft_conv_left, 
        bottomleft_conv_left, 
        topright_conv_left, 
        bottomright_conv_left
    ])


    l_merge_right = ElemwiseSumLayer([
        topleft_conv_right, 
        bottomleft_conv_right, 
        topright_conv_right, 
        bottomright_conv_right        
    ])


    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        l_merge_left = DropoutLayer(l_merge_left, p=args["end_dropout"])
        l_merge_right = DropoutLayer(l_merge_right, p=args["end_dropout"])

    
    #l_dropout = DropoutLayer(l_merge, p=args["fc_p"])

    # left out
    l_merge_left_sum = ElemwiseSumLayer([
        l_merge_left,
        l_merge_right
    ])
    l_out_left = DenseLayer(
        l_merge_left_sum,
        num_units=5,
        nonlinearity=softmax,
        W=HeNormal()
    )

    # 22/07/2016: fix weight/bias sharing between left and right softmax
    # layers

    # right out
    l_merge_right_sum = ElemwiseSumLayer([
        l_merge_left,
        l_merge_right
    ])

    if args["fix_softmax"]:
        sys.stderr.write("fix softmax bug...\n")
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=l_out_left.W,
            b=l_out_left.b
        )
    else:
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=HeNormal()
        )        

    l_out_pseudo = ElemwiseSumLayer([l_out_left, l_out_right])
    
    
    
    #l_out = lasagne.layers.DenseLayer(
    #    l_dropout,
    #    num_units=5,
    #    nonlinearity=softmax,
    #    W=HeNormal()
    #)

    sys.stderr.write("number of params: %i\n" % count_params(l_out_pseudo))
    
    return {"l_out_pseudo": l_out_pseudo,
            "l_out_left": l_out_left,
            "l_out_right": l_out_right,
            "l_in_left": l_in_left,
            "l_in_right": l_in_right}







def resnet_net_beefier(args={}):
    
    #conv_p, fc_p = args["fc_p"]
    
    l_in_left = InputLayer( (None, 3, 512, 512) )
    l_in_right = InputLayer( (None, 3, 512, 512) )
    # left image
    l_topleft_left = SliceLayer( SliceLayer(l_in_left, indices=slice(0,256), axis=2), indices=slice(0,256), axis=3 )
    l_bottomleft_left = SliceLayer( SliceLayer(l_in_left, indices=slice(256,512), axis=2), indices=slice(0,256), axis=3 )
    l_topright_left = SliceLayer( SliceLayer(l_in_left, indices=slice(0,256), axis=2), indices=slice(256,512), axis=3 )
    l_bottomright_left = SliceLayer( SliceLayer(l_in_left, indices=slice(256,512), axis=2), indices=slice(256,512), axis=3 )
    # right image
    l_topleft_right = SliceLayer( SliceLayer(l_in_right, indices=slice(0,256), axis=2), indices=slice(0,256), axis=3 )
    l_bottomleft_right = SliceLayer( SliceLayer(l_in_right, indices=slice(256,512), axis=2), indices=slice(0,256), axis=3 )
    l_topright_right = SliceLayer( SliceLayer(l_in_right, indices=slice(0,256), axis=2), indices=slice(256,512), axis=3 )
    l_bottomright_right = SliceLayer( SliceLayer(l_in_right, indices=slice(256,512), axis=2), indices=slice(256,512), axis=3 ) 

    if "dropout_p" in args:
        dropout_p = args["dropout_p"]
        sys.stderr.write("dropout_p: %f\n" % dropout_p)
    else:
        dropout_p = None
    
    dd, topleft_conv_left = resnet(l_topleft_left, {}, True, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    bottomleft_conv_left = resnet(l_bottomleft_left, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    topright_conv_left = resnet(l_topright_left, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    bottomright_conv_left = resnet(l_bottomright_left, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    
    topleft_conv_right = resnet(l_topleft_right, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    bottomleft_conv_right = resnet(l_bottomleft_right, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    topright_conv_right = resnet(l_topright_right, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    bottomright_conv_right = resnet(l_bottomright_right, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)

    for layer in get_all_layers(topleft_conv_left):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    l_merge_left = ElemwiseSumLayer([
        topleft_conv_left, 
        bottomleft_conv_left, 
        topright_conv_left, 
        bottomright_conv_left
    ])


    l_merge_right = ElemwiseSumLayer([
        topleft_conv_right, 
        bottomleft_conv_right, 
        topright_conv_right, 
        bottomright_conv_right        
    ])


    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        l_merge_left = DropoutLayer(l_merge_left, p=args["end_dropout"])
        l_merge_right = DropoutLayer(l_merge_right, p=args["end_dropout"])

    
    #l_dropout = DropoutLayer(l_merge, p=args["fc_p"])

    # left out
    l_merge_left_sum = ElemwiseSumLayer([
        l_merge_left,
        l_merge_right
    ])
    l_out_left = DenseLayer(
        l_merge_left_sum,
        num_units=5,
        nonlinearity=softmax,
        W=HeNormal()
    )

    # 22/07/2016: fix weight/bias sharing between left and right softmax
    # layers

    # right out
    l_merge_right_sum = ElemwiseSumLayer([
        l_merge_left,
        l_merge_right
    ])

    if args["fix_softmax"]:
        sys.stderr.write("fix softmax bug...\n")
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=l_out_left.W,
            b=l_out_left.b
        )
    else:
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=HeNormal()
        )        

    l_out_pseudo = ElemwiseSumLayer([l_out_left, l_out_right])
    
    sys.stderr.write("number of params: %i\n" % count_params(l_out_pseudo))
    
    return {"l_out_pseudo": l_out_pseudo,
            "l_out_left": l_out_left,
            "l_out_right": l_out_right,
            "l_in_left": l_in_left,
            "l_in_right": l_in_right}


















def resnet_net_256(args={}):
    
    l_in_left = InputLayer( (None, 3, 256, 256) )
    l_in_right = InputLayer( (None, 3, 256, 256) )

    dd, conv_left = resnet(l_in_left, {}, True)
    conv_right = resnet(l_in_right, dd, False)

    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv_left = DropoutLayer(conv_left, p=args["end_dropout"])
        conv_right = DropoutLayer(conv_right, p=args["end_dropout"])
    
    for layer in get_all_layers(conv_left):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")

    # left out
    l_merge_left_sum = ElemwiseSumLayer([
        conv_left,
        conv_right
    ])
    l_out_left = DenseLayer(
        l_merge_left_sum,
        num_units=5,
        nonlinearity=softmax,
        W=HeNormal()
    )

    # 22/07/2016 -- fix weight sharing bug here with softmax layers

    # right out
    l_merge_right_sum = ElemwiseSumLayer([
        conv_left,
        conv_right
    ])

    if args["fix_softmax"]:
        sys.stderr.write("fix softmax bug...\n")
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=l_out_left.W,
            b=l_out_left.b
        )
    else:
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=HeNormal()
        )        

    
    l_out_pseudo = ElemwiseSumLayer([l_out_left, l_out_right])

    sys.stderr.write("number of params: %i\n" % count_params(l_out_pseudo))
    
    return {"l_out_pseudo": l_out_pseudo,
            "l_out_left": l_out_left,
            "l_out_right": l_out_right,
            "l_in_left": l_in_left,
            "l_in_right": l_in_right}



def resnet_net_256_beefier(args={}):
    
    l_in_left = InputLayer( (None, 3, 256, 256) )
    l_in_right = InputLayer( (None, 3, 256, 256) )

    if "dropout_p" in args:
        dropout_p = args["dropout_p"]
        sys.stderr.write("dropout_p: %f\n" % dropout_p)
    else:
        dropout_p = None
    
    dd, conv_left = resnet(l_in_left, {}, True, nf=[64, 128, 256, 512], dropout_p=dropout_p)
    conv_right = resnet(l_in_right, dd, False, nf=[64, 128, 256, 512], dropout_p=dropout_p)

    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv_left = DropoutLayer(conv_left, p=args["end_dropout"])
        conv_right = DropoutLayer(conv_right, p=args["end_dropout"])
    
    for layer in get_all_layers(conv_left):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")

    # left out
    l_merge_left_sum = ElemwiseSumLayer([
        conv_left,
        conv_right
    ])
    l_out_left = DenseLayer(
        l_merge_left_sum,
        num_units=5,
        nonlinearity=softmax,
        W=HeNormal()
    )

    # 22/07/2016 -- fix weight sharing bug here with softmax layers

    # right out
    l_merge_right_sum = ElemwiseSumLayer([
        conv_left,
        conv_right
    ])

    if args["fix_softmax"]:
        sys.stderr.write("fix softmax bug...\n")
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=l_out_left.W,
            b=l_out_left.b
        )
    else:
        l_out_right = DenseLayer(
            l_merge_right_sum,
            num_units=5,
            nonlinearity=softmax,
            W=HeNormal()
        )        

    
    l_out_pseudo = ElemwiseSumLayer([l_out_left, l_out_right])

    sys.stderr.write("number of params: %i\n" % count_params(l_out_pseudo))
    
    return {"l_out_pseudo": l_out_pseudo,
            "l_out_left": l_out_left,
            "l_out_right": l_out_right,
            "l_in_left": l_in_left,
            "l_in_right": l_in_right}

















# In[11]:

imgen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=359.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.,
    zoom_range=0.02,
    channel_shift_range=0.,
    fill_mode='constant',
    cval=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None)


def load_image_keras(filename):
    img = io.imread(filename)
    img = img_as_float(img)
    img = np.asarray( [ img[...,0], img[...,1], img[...,2] ] ) # reshape
    for i in range(0, img.shape[0]):
        img[i, ...] = (img[i, ...] - np.mean(img[i, ...])) / np.std(img[i,...]) # zmuv
    for xb, _ in imgen.flow( np.asarray([img], dtype=img.dtype), np.asarray([0], dtype="int32")):
        break
    return xb[0]
    

def iterate(X_arr_left, X_arr_right, y_arr_left, y_arr_right, bs, augment, zmuv, keras=False, DATA_DIR=os.environ["DATA_DIR"]):
    assert X_arr_left.shape[0] == X_arr_right.shape[0] == y_arr_left.shape[0] == y_arr_right.shape[0]
    b = 0
    #DATA_DIR = os.environ["DATA_DIR"]
    while True:
        if b*bs >= X_arr_left.shape[0]:
            break
        this_X_left, this_X_right, this_y_left, this_y_right = \
                        X_arr_left[b*bs:(b+1)*bs], X_arr_right[b*bs:(b+1)*bs], y_arr_left[b*bs:(b+1)*bs], y_arr_right[b*bs:(b+1)*bs]

        if not keras:
            images_for_this_X_left = [ hp.load_image_fast("%s/%s.jpeg" % (DATA_DIR,filename), augment=augment, zmuv=zmuv) for filename in this_X_left ]
        else:
            images_for_this_X_left = [ load_image_keras("%s/%s.jpeg" % (DATA_DIR,filename) ) for filename in this_X_left ]
        images_for_this_X_left = np.asarray(images_for_this_X_left, dtype="float32")

        if not keras:
            images_for_this_X_right = [ hp.load_image_fast("%s/%s.jpeg" % (DATA_DIR,filename), augment=augment, zmuv=zmuv) for filename in this_X_right ]
        else:
            images_for_this_X_right = [ load_image_keras("%s/%s.jpeg" % (DATA_DIR,filename) ) for filename in this_X_right ]
        images_for_this_X_right = np.asarray(images_for_this_X_right, dtype="float32")

        #print images_for_this_X_right.shape
        #print images_for_this_X_left.shape
        #print this_X_left.shape
        #print this_X_right.shape
        
        #print images_for_this_X_left.shape
        #print images_for_this_X_right.shape
        #print this_y.shape

        yield images_for_this_X_left, images_for_this_X_right, this_y_left, this_y_right
        
        # ---
        b += 1

        
# In[19]:

def train(net_cfg, 
          num_epochs,
          data,
          out_file=None,
          print_out=True,
          debug=False,
          resume=None,
          augment=True,
          zmuv=False,
          keras=False,
          schedule={}):
    # prepare the out_file
    l_out_left = net_cfg["l_out_left"]
    l_out_right = net_cfg["l_out_right"]
    l_out_pseudo = net_cfg["l_out_pseudo"]
    
    f = None
    if resume == None:
        if out_file != None:
            f = open("%s.txt" % out_file, "wb")
            f.write("epoch,train_loss,avg_valid_loss,valid_accuracy,valid_kappa,valid_kappa_exp,time\n")
        if print_out:
            print "epoch,train_loss,avg_valid_loss,valid_accuracy,valid_kappa,valid_kappa_exp,time"
    else:
        sys.stderr.write("resuming training...\n")
        if out_file != None:
            f = open("%s.txt" % out_file, "ab")
        with open(resume) as g:
            set_all_param_values(l_out_pseudo, pickle.load(g))

        
            
    # save graph of network
    #draw_net.draw_to_file( get_all_layers(l_out), "%s.png" % out_file, verbose=True)
    # extract functions
    X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right = data
    train_fn, loss_fn, preds_fn, dist_fn = net_cfg["train_fn"], net_cfg["loss_fn"], net_cfg["preds_fn"], net_cfg["dist_fn"]
    learning_rate = net_cfg["learning_rate"]
    bs = net_cfg["bs"]
    
    # training
    train_idxs = [x for x in range(0, X_train_left.shape[0])]
    
    if debug:
        sys.stderr.write("idxs: %s\n" % train_idxs)
    for epoch in range(0, num_epochs):
        
        if epoch+1 in schedule:
            sys.stderr.write("changing learning rate to: %f" % schedule[epoch+1])
            learning_rate.set_value( floatX(schedule[epoch+1]) )
        
        np.random.shuffle(train_idxs)
        X_train_left, X_train_right = X_train_left[train_idxs], X_train_right[train_idxs]
        y_train_left, y_train_right = y_train_left[train_idxs], y_train_right[train_idxs]
        
        # training loop
        this_train_losses = []
        t0 = time()
        for X_train_batch_left, X_train_batch_right, y_train_batch_left, y_train_batch_right in \
                        iterate(X_train_left, X_train_right, y_train_left, y_train_right, bs, augment, zmuv, keras):
            this_train_losses.append( train_fn(X_train_batch_left, X_train_batch_right, y_train_batch_left, y_train_batch_right) )
        time_taken = time() - t0
        
        # validation loss loop
        this_valid_losses = []
        for X_valid_batch_left, X_valid_batch_right, y_valid_batch_left, y_valid_batch_right in\
                        iterate(X_valid_left, X_valid_right, y_valid_left, y_valid_right, bs, False, zmuv, keras):
            this_valid_losses.append( loss_fn(X_valid_batch_left, X_valid_batch_right, y_valid_batch_left, y_valid_batch_right) )
        avg_valid_loss = np.mean(this_valid_losses)
        
        # validation accuracy loop
        left_valid_preds = []
        right_valid_preds = []

        left_valid_preds_exp = []
        right_valid_preds_exp = []
        
        for X_valid_batch_left, X_valid_batch_right, _, _ in iterate(X_valid_left, X_valid_right, y_valid_left, y_valid_right, bs, False, zmuv, keras):
            # just do argmax to get the predictions for valid_kappa
            left_preds, right_preds = preds_fn(X_valid_batch_left, X_valid_batch_right)
            left_valid_preds += left_preds.tolist()
            right_valid_preds += right_preds.tolist()

            # compute the valid_kappa_exp
            left_dist, right_dist = dist_fn(X_valid_batch_left, X_valid_batch_right)
            for dist in left_dist:
                left_valid_preds_exp.append( int(np.round(np.dot(dist, np.arange(0,5)))) )
            for dist in right_dist:
                right_valid_preds_exp.append( int(np.round(np.dot(dist, np.arange(0,5)))) )
            
            
        total_valid_preds = np.hstack((left_valid_preds, right_valid_preds))
        total_valid_y = np.hstack((y_valid_left, y_valid_right))
        valid_acc = np.mean(total_valid_preds == total_valid_y)

        total_valid_exp_preds = np.hstack((left_valid_preds_exp, right_valid_preds_exp))
        
        # validation set kappa
        valid_kappa = hp.weighted_kappa(human_rater=total_valid_preds, actual_rater=total_valid_y, num_classes=5)

        valid_kappa_exp = hp.weighted_kappa(human_rater=total_valid_exp_preds, actual_rater=total_valid_y, num_classes=5)
        
        ## ------------ ##
        if f != None:
            f.write(
                "%i,%f,%f,%f,%f,%f,%f\n" %
                    (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, valid_kappa, valid_kappa_exp, time_taken) 
            )
            f.flush()
        if print_out:
            print "%i,%f,%f,%f,%f,%f,%f" % (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, valid_kappa, valid_kappa_exp, time_taken)
            
        with open("models/%s.model.%i" % (os.path.basename(out_file),epoch+1), "wb") as g:
            pickle.dump(get_all_param_values(l_out_pseudo), g, pickle.HIGHEST_PROTOCOL) 
            
    if f != None:
        f.close()


if __name__ == "__main__":



    # train without augmentation and without dropout
    if "EXP1_NO_AUGMENT" in os.environ:
        seed = 0
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(net, {"kappa_loss": False, "batch_size": 32, "conv_p": 0.0, "fc_p": 0.0 }) 
        train(
            cfg, 
            num_epochs=1000, 
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/exp1_no-augment.%i" % seed,
            augment=False
        )

    # train with dropout, don't do augmentation yet
    if "EXP1" in os.environ:
        seed = 0
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(net, {"kappa_loss": False, "batch_size": 32, "conv_p": 0.1, "fc_p": 0.5 }) 
        train(
            cfg, 
            num_epochs=1000, 
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/exp1_no-augment-with-dropout.%i" % seed,
            augment=False,
            resume="models/exp1_no-augment-with-dropout.0.model.1"
        )






    # train with dropout, do augmentation
    if "EXP1_AUGMENT_AND_DROPOUT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(net, {"kappa_loss": False, "batch_size": 64, "conv_p": 0.1, "fc_p": 0.5 }) 
        train(
            cfg, 
            num_epochs=1000, 
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/exp1_augment-with-dropout.%i" % seed,
            augment=True
        )

    if "EXP1_AUGMENT_AND_DROPOUT_LESS" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(net, {"kappa_loss": False, "batch_size": 64, "conv_p": 0.1, "fc_p": 0.1 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/exp1_augment-with-dropout-0.1both.%i" % seed,
            augment=True,
            resume="models/exp1_augment-with-dropout-0.1both.1.model.2"
        )



    # -----------------

    if "EXP1_ONLY_AUGMENT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(net, {"kappa_loss": False, "batch_size": 64, "fc_p": 0.0 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/exp1_only_augment.%i" % seed,
            augment=True,
            resume="models/exp1_only_augment.1.model.16.bak"
        )


    if "EXP1_AUGMENT_AND_DROPOUT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(net, {"kappa_loss": False, "batch_size": 64, "fc_p": 0.5 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/exp1_d0.5.%i" % seed,
            augment=True
        )

    # ----------    

    if "RESNET_ONLY_AUGMENT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 48, "fc_p": 0.0 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/resnet_only_augment.%i" % seed,
            augment=True,
            resume="models/resnet_only_augment.1.model.16"
        )


    if "RESNET_ONLY_AUGMENT_64" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "fc_p": 0.0 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/resnet_only_augment_b64.%i" % seed,
            augment=True
        )



    if "RESNET_ONLY_AUGMENT_64_ZMUV" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "fc_p": 0.0 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/resnet_only_augment_b64_zmuv.%i" % seed,
            augment=True,
            zmuv=True
        )


    if "RESNET_ONLY_AUGMENT_64_ZMUV_L2" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "fc_p": 0.0, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train, X_valid_left, X_valid_right, y_valid),
            out_file="output_quadrant/resnet_only_augment_b64_zmuv_l2.%i" % seed,
            augment=True,
            zmuv=True
        )

    # -- fixed l2 bug and do left/right classes


    if "NEW_ONLY_AUGMENT_64_ZMUV_L2" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2.%i" % seed,
            augment=True,
            zmuv=True
        )                        


    if "TEST_LR" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "learning_rate":0.1 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_lr0.1.%i" % seed,
            augment=True,
            zmuv=True
        )                        





    if "RESUME_LLR" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "learning_rate":0.001  })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_llr.%i" % seed,
            augment=True,
            zmuv=True,
            resume="models/new_only_augment_b64_zmuv_l2.1.model.56.bak"
        )

    if "RESUME_MORE_AUGMENT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_more-aug.%i" % seed,
            augment=True,
            zmuv=True,
            resume="models/new_only_augment_b64_zmuv_l2.1.model.56.bak"
        )



    if "RESUME_MORE_AUGMENT_LLR" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "learning_rate": 0.001 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_more-aug_lr1e-3.%i" % seed,
            augment=True,
            zmuv=True,
            resume="models/new_only_augment_b64_zmuv_l2_more-aug.1.model.103.bak"
        )



    if "RESUME_MORE_AUGMENT_ABSORB" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_more-aug_absorb.%i" % seed,
            augment=True,
            zmuv=True,
            resume="models/new_only_augment_b64_zmuv_l2_more-aug.1.model.103.bak"
        )



    # try experimenting with keras' data augmentation
    # - is it faster?
    # - added shift width/height augment
    # - added slight zoom augment
    # - vertical + horizontal flipping (in addition to rotation which we already have)
    if "RESUME_KERAS_AUGMENT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_keras-aug.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/new_only_augment_b64_zmuv_l2.1.model.56.bak"
        )

    if "RESUME_KERAS_AUGMENT_ABSORB_AND_FSM" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_keras-aug_absorb_fsm.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/new_only_augment_b64_zmuv_l2_keras-aug.1.model.100.bak.fixsoftmax"
        )

    if "RESUME_KERAS_AUGMENT_ABSORB_AND_FSM_AND_DROPOUT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True, "end_dropout":0.5 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_keras-aug_absorb_fsm_d0.5.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/new_only_augment_b64_zmuv_l2_keras-aug.1.model.100.bak.fixsoftmax"
        )

    if "RESUME_KERAS_AUGMENT_ABSORB_AND_FSM_AND_DROPOUT_AND_CONV_DROPOUT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True, "end_dropout":0.5, "dropout_p": 0.1 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_keras-aug_absorb_fsm_d0.5_convd0.1.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/new_only_augment_b64_zmuv_l2_keras-aug_absorb_fsm_d0.5.1.model.91.bak"
        )


    # ----------

    if "RESNET_BEEFY" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_beefier, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True, "end_dropout":0.5 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/resnet-beefy_absorb_fsm_d0.5.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )





    # -----------

    if "KERAS_FSM_NO_BEN" in os.environ:
        seed = 1
        os.environ["DATA_DIR"] = "/tmp/beckhamc/train-trim-512/"
        sys.stderr.write("using data folder: %s\n" % os.environ["DATA_DIR"])
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/keras_fsm_no_ben.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )



    # ------------    

    if "LOW_RES" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )
    if "LOW_RES_RESUME_ABSORB" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/low_res.1.model.137.bak"
        )
    if "LOW_RES_RESUME_ABSORB_FSM" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/low_res.1.model.3.bak2.fixsoftmax"
        )




    if "LOW_RES_FSM_BEEFIER" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256_beefier, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_fsm_beefier.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )
    if "LOW_RES_FSM_BEEFIER_DROPOUT" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256_beefier, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True, "end_dropout":0.5 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_fsm_beefier.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/low_res_fsm_beefier.1.model.29.bak"
        )
    if "LOW_RES_FSM_BEEFIER_DROPOUT_RESUME" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256_beefier, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax":True, "end_dropout":0.5, "dropout_p":0.4 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_fsm_beefier.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
            resume="models/low_res_fsm_beefier.1.model.101.bak2"
        )








    if "LOW_RES_FSM" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "fix_softmax": True })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_fsm.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )




    if "LOW_RES_5PVALID" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4 })
        # change the dataset so that 5% is valid set, not 10%
        X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
        X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
        y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
        y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
        X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
        X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
        y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
        y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_5pvalid.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True,
        )




    if "LOW_RES_DROPOUT_FSM" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": False, "batch_size": 64, "l2": 1e-4, "end_dropout":0.5, "fix_softmax":True })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_d0.5_fsm.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )


    # this has really slow convergence and doesn't boost the valid kappa
    # so let's try a hybrid loss function
    if "LOW_RES_KAPPA" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": True, "batch_size": 64, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_kappa.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )


    if "LOW_RES_KAPPA_HYBRID_01" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": True, "hybrid_loss":0.01, "batch_size": 64, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_kappa_hybrid_01.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )


    if "LOW_RES_KAPPA_HYBRID_001" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net_256, { "kappa_loss": True, "hybrid_loss":0.001, "batch_size": 64, "l2": 1e-4 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/low_res_kappa_hybrid_001.%i" % seed,
            augment=True,
            zmuv=True,
            keras=True
        )




    # ----



    if "NEW_ONLY_AUGMENT_64_ZMUV_L2_KAPPA" in os.environ:
        seed = 1
        lasagne.random.set_rng( np.random.RandomState(seed) )
        cfg = get_net(resnet_net, { "kappa_loss": True, "batch_size": 64, "l2": 1e-4, "learning_rate": 0.1 })
        train(
            cfg,
            num_epochs=1000,
            data=(X_train_left, X_train_right, y_train_left, y_train_right, X_valid_left, X_valid_right, y_valid_left, y_valid_right),
            out_file="output_quadrant/new_only_augment_b64_zmuv_l2_kappa.%i" % seed,
            augment=True,
            zmuv=True
        )                        
