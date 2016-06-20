
# coding: utf-8

# todo: move to lasagne recipes fork, and import the cifar10 data loader, and run exps on cuda4

# In[7]:

import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.regularization import *
from lasagne.random import get_rng
from lasagne.updates import *
from lasagne.init import *
import numpy as np
import sys
sys.setrecursionlimit(10000)
sys.path.append("../../modules/")
import helper as hp
from lasagne.utils import floatX

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

import os
import cPickle as pickle

from theano.tensor import TensorType

from theano.ifelse import ifelse

from time import time

#get_ipython().magic(u'load_ext rpy2.ipython')

from scipy import stats

import deep_residual_learning_CIFAR10

import math


# We keep all the contents of the tensor with survival probability $p$, so the expectation at test time is also $p$.

# In[2]:

class BinomialDropLayer(Layer):
    def __init__(self, incoming, p=0.5, **kwargs):
        super(BinomialDropLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return self.p*input
        else:
            #mask = self._srng.binomial(n=1, p=(self.p), size=(input.shape[0],),
            #    dtype=input.dtype)
            # apply the same thing to all examples in the minibatch
            mask = T.zeros((input.shape[0],)) + self._srng.binomial((1,), p=self.p, dtype=input.dtype)[0]
            mask = mask.dimshuffle(0,'x','x','x')
            return mask*input


# In[ ]:

class IfElseDropLayer(Layer):
    def __init__(self, incoming, p=0.5, **kwargs):
        super(IfElseDropLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return self.p*input
        else:
            return ifelse(
                T.lt(self._srng.uniform( (1,), 0, 1)[0], self.p),
                input,
                T.zeros(input.shape)
            )


# In[ ]:

class SkippableNonlinearityLayer(Layer):
    def __init__(self, incoming, nonlinearity=rectify, p=0.5, **kwargs):
        super(SkippableNonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0.0:
            # apply the bernoulli expectation
            return self.p*input + (1-self.p)*self.nonlinearity(input)
        else:
            if input.ndim==4:
                mask = self._srng.binomial(n=1, p=(self.p), size=(input.shape[0],1,1,1),
                    dtype=input.dtype)
                mask = T.addbroadcast(mask, 1,2,3)
                return mask*input + (1-mask)*self.nonlinearity(input)
            elif input.ndim == 2:
                mask = self._srng.binomial(n=1, p=(self.p), size=(input.shape[0],1),
                    dtype=input.dtype)
                mask = T.addbroadcast(mask, 1)
                return mask*input + (1-mask)*self.nonlinearity(input) 


# In[ ]:

class MoreSkippableNonlinearityLayer(Layer):
    def __init__(self, incoming, nonlinearity=rectify, p=0.5,
                 **kwargs):
        super(MoreSkippableNonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0.0:
            # apply the bernoulli expectation
            return self.p*input + (1-self.p)*self.nonlinearity(input)
        else:
            mask = self._srng.binomial(n=1, p=(self.p), size=input.shape,
                dtype=input.dtype)
            return mask*input + (1-mask)*self.nonlinearity(input) 


# There is a difference between this residual block method and the one that is defined in [link]. When the number of filters is different to the layer's output shape (or the stride is different), instead of using a convolution to make things compatible, we use an average pooling with a pool size of 1 and a the defined stride, followed by (if necessary) adding extra zero-padded feature maps. This is because this is how the authors in [link] have defined it.

# In[32]:

def get_init(name):
    if name == "glorot":
        return GlorotUniform()
    else:
        print "yah"
        return HeNormal(gain="relu")


# In[12]:

def residual_block(layer, n_out_channels, stride=1, survival_p=None, nonlinearity_p=None, args={}):
    conv = layer
    if "nonlinearity" in args:
        nonlinearity = args["nonlinearity"]
        sys.stderr.write("using nonlinearity %s\n" % str(args["nonlinearity"]))
    else:
        nonlinearity = rectify
    if stride > 1:
        layer = Pool2DLayer(layer, pool_size=1, stride=stride, mode="average_inc_pad")
    if (n_out_channels != layer.output_shape[1]):
        diff = n_out_channels-layer.output_shape[1]
        if diff % 2 == 0: 
            width_tp = ((diff/2, diff/2),)
        else:
            width_tp = (((diff/2)+1, diff/2),)
        layer = pad(layer, batch_ndim=1, width=width_tp)
    conv = Conv2DLayer(conv, num_filters=n_out_channels,
                       filter_size=(3,3), stride=(stride,stride), pad=(1,1), nonlinearity=linear, W=get_init(args["init"]))
    conv = BatchNormLayer(conv)
    if nonlinearity_p == None:
        conv = NonlinearityLayer(conv, nonlinearity=nonlinearity)
    else:
        conv = MoreSkippableNonlinearityLayer(conv, p=nonlinearity_p, nonlinearity=nonlinearity)
    conv = Conv2DLayer(conv, num_filters=n_out_channels,
                       filter_size=(3,3), stride=(1,1), pad=(1,1), nonlinearity=linear, W=get_init(args["init"]))
    conv = BatchNormLayer(conv)
    if survival_p != None:
        conv = BinomialDropLayer(conv, p=survival_p)
    if nonlinearity_p == None:
        return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity=nonlinearity)
    else:
        return MoreSkippableNonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity=nonlinearity)


# In[15]:

def yu_cifar10_net(args):
    # Architecture from:
    # https://github.com/yueatsprograms/Stochastic_Depth/blob/master/main.lua
    N = 18
    survival_p = args["survival_p"]
    nonlinearity_p = args["nonlinearity_p"]
    layer = InputLayer( (None, 3, 32, 32) )
    if "nonlinearity" in args:
        nonlinearity = args["nonlinearity"]
        sys.stderr.write("using nonlinearity %s\n" % str(args["nonlinearity"]))
    else:
        nonlinearity = rectify
    layer = Conv2DLayer(layer, num_filters=16, filter_size=3, stride=1, nonlinearity=nonlinearity, pad='same', W=get_init(args["init"]))
    #layer = Pool2DLayer(layer, 2)
    for _ in range(N):
        layer = residual_block(layer, 16, survival_p=survival_p, nonlinearity_p=args["nonlinearity_p"], args=args)
    layer = residual_block(layer, 32, stride=2, survival_p=survival_p, nonlinearity_p=args["nonlinearity_p"], args=args)
    for _ in range(N):
        layer = residual_block(layer, 32, survival_p=survival_p, nonlinearity_p=args["nonlinearity_p"], args=args)
    layer = residual_block(layer, 64, stride=2, survival_p=survival_p, nonlinearity_p=args["nonlinearity_p"], args=args)
    for _ in range(N):
        layer = residual_block(layer, 64, survival_p=survival_p, nonlinearity_p=args["nonlinearity_p"], args=args)
    layer = Pool2DLayer(layer, pool_size=8, stride=1, mode="average_inc_pad")
    layer = DenseLayer(layer, num_units=10, nonlinearity=softmax,
                       W=GlorotUniform() if args["init"] == "glorot" else HeNormal())
    for layer in get_all_layers(layer):
        print layer, layer.output_shape
    print "number of params:", count_params(layer)
    return layer


# In[16]:

def linear_decay(l, L, pL=0.5):
    assert l <= L
    l = l*1.0
    L = L*1.0
    return 1.0 - ((l/L)*(1-pL))


# In[17]:

def yu_cifar10_net_decay(args):
    # Architecture from:
    # https://github.com/yueatsprograms/Stochastic_Depth/blob/master/main.lua
    
    N = 18
    layer = InputLayer( (None, 3, 32, 32) )
    if "nonlinearity" in args:
        nonlinearity = args["nonlinearity"]
    else:
        nonlinearity = rectify
    layer = Conv2DLayer(layer, num_filters=16, filter_size=3, stride=1, nonlinearity=nonlinearity, pad='same') #BUG
    #layer = Pool2DLayer(layer, 2)
    l = 1
    L = 3*N + 2
    for _ in range(N):
        if args["decay"] == "depth":
            layer = residual_block(layer, 16, survival_p=linear_decay(l,L), nonlinearity_p=None, args=args)
        elif args["decay"] == "nonlinearity":
            layer = residual_block(layer, 16, survival_p=None, nonlinearity_p=linear_decay(l,L),args=args)
        elif args["decay"] == "both":
            layer = residual_block(layer, 16, survival_p=linear_decay(l,L), nonlinearity_p=linear_decay(l,L),args=args)
        #print linear_decay(l,L)
        l += 1
    if args["decay"] == "depth":
        layer = residual_block(layer, 32, stride=2, survival_p=linear_decay(l,L), nonlinearity_p=None,args=args)
    elif args["decay"] == "nonlinearity":
        layer = residual_block(layer, 32, stride=2, survival_p=None, nonlinearity_p=linear_decay(l,L),args=args)
    elif args["decay"] == "both":
        layer = residual_block(layer, 32, stride=2, survival_p=linear_decay(l,L), nonlinearity_p=linear_decay(l,L),args=args)
    #print linear_decay(l,L)
    l += 1
    for _ in range(N):
        if args["decay"] == "depth":
            layer = residual_block(layer, 32, survival_p=linear_decay(l,L), nonlinearity_p=None,args=args)
        elif args["decay"] == "nonlinearity":
            layer = residual_block(layer, 32, survival_p=None, nonlinearity_p=linear_decay(l,L), args=args)
        elif args["decay"] == "both":
            layer = residual_block(layer, 32, survival_p=linear_decay(l,L), nonlinearity_p=linear_decay(l,L), args=args)
        #print linear_decay(l,L)
        l += 1
    if args["decay"] == "depth":
        layer = residual_block(layer, 64, stride=2, survival_p=linear_decay(l,L), nonlinearity_p=None, args=args)
    elif args["decay"] == "nonlinearity":
        layer = residual_block(layer, 64, stride=2, survival_p=None, nonlinearity_p=linear_decay(l,L), args=args)
    elif args["decay"] == "both":
        layer = residual_block(layer, 64, stride=2, survival_p=linear_decay(l,L), nonlinearity_p=linear_decay(l,L), args=args)
    #print linear_decay(l,L)
    l += 1
    for _ in range(N):
        if args["decay"] == "depth":
            layer = residual_block(layer, 64, survival_p=linear_decay(l,L), nonlinearity_p=None, args=args)
        elif args["decay"] == "nonlinearity":
            layer = residual_block(layer, 64, survival_p=None, nonlinearity_p=linear_decay(l,L), args=args)
        elif args["decay"] == "both":
            layer = residual_block(layer, 64, survival_p=linear_decay(l,L), nonlinearity_p=linear_decay(l,L), args=args)
        #print linear_decay(l,L)
        l += 1
    #print "l, L =", l, L
    layer = Pool2DLayer(layer, pool_size=8, stride=1, mode="average_inc_pad")
    layer = DenseLayer(layer, num_units=10, nonlinearity=softmax, W=GlorotUniform() if args["init"] == "glorot" else HeNormal())
    for layer in get_all_layers(layer):
        print layer, layer.output_shape
    print "number of params:", count_params(layer)
    return layer


# In[16]:

def debug_net(args):
    layer = InputLayer( (None, 3, 32, 32) )
    layer = Conv2DLayer(layer, num_filters=8, filter_size=3)
    layer = MaxPool2DLayer(layer, pool_size=2)
    layer = Conv2DLayer(layer, num_filters=8, filter_size=3)
    layer = MaxPool2DLayer(layer, pool_size=2)
    layer = Conv2DLayer(layer, num_filters=8, filter_size=3)
    layer = MaxPool2DLayer(layer, pool_size=2)
    layer = DenseLayer(layer, nonlinearity=softmax, num_units=10)
    return layer


# ----

# In[4]:

"""
data = deep_residual_learning_CIFAR10.load_data()
if "QUICK" in os.environ:
    sys.stderr.write("loading smaller version of cifar10...\n")
    X_train_and_valid, y_train_and_valid, X_test, y_test = \
        data["X_train"][0:500], data["Y_train"][0:500], data["X_test"][0:500], data["Y_test"][0:500]
else:
    if "AUGMENT" not in os.environ:
        X_train_and_valid, y_train_and_valid, X_test, y_test = \
            data["X_train"][0:50000], data["Y_train"][0:50000], data["X_test"], data["Y_test"]
    else:
        sys.stderr.write("data augmentation on...\n")
        X_train_and_valid, y_train_and_valid, X_test, y_test = \
            data["X_train"], data["Y_train"], data["X_test"], data["Y_test"]
"""


# In[5]:

"""
X_train = X_train_and_valid[ 0 : 0.9*X_train_and_valid.shape[0] ]
y_train = y_train_and_valid[ 0 : 0.9*y_train_and_valid.shape[0] ]
X_valid = X_train_and_valid[ 0.9*X_train_and_valid.shape[0] :: ]
y_valid = y_train_and_valid[ 0.9*y_train_and_valid.shape[0] :: ]
"""


# In[2]:

"""
X_train = theano.shared(np.asarray(X_train, dtype=theano.config.floatX), borrow=True)
y_train = theano.shared(np.asarray(y_train, dtype=theano.config.floatX), borrow=True)
X_valid = theano.shared(np.asarray(X_valid, dtype=theano.config.floatX), borrow=True)
y_valid = theano.shared(np.asarray(y_valid, dtype=theano.config.floatX), borrow=True)
X_test = theano.shared(np.asarray(X_test, dtype=theano.config.floatX), borrow=True)
y_test = theano.shared(np.asarray(y_test, dtype=theano.config.floatX), borrow=True)
"""


# In[8]:

dat = np.load("cifar10.npz")
X_train, y_train, X_valid, y_valid = dat["X_train"], dat["y_train"], dat["X_valid"], dat["y_valid"]
X_train = X_train.astype("float32")
y_train = y_train.astype("int32")
X_valid = X_valid.astype("float32")
y_valid = y_valid.astype("int32")


# In[9]:

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# -----

# In[22]:

def get_batch_idxs(x, bs):
    b = 0
    arr = []
    while True:
        #print b, x[b*bs : (b+1)*bs].shape
        if x[b*bs : (b+1)*bs].shape[0] != 0:
            arr.append(b)
        else:
            break
        b += 1
    return arr


# In[10]:

def get_net(l_out, data, args={}):
    # ----
    X = T.tensor4('X')
    y = T.ivector('y')
    idx = T.lscalar('idx')
    # ----
    net_out = get_output(l_out, X)
    net_out_det = get_output(l_out, X, deterministic=True)
    loss = categorical_crossentropy(net_out, y).mean()
    loss_det = categorical_crossentropy(net_out_det, y).mean()
    if "l2" in args:
        sys.stderr.write("adding l2: %f\n" % args["l2"])
        loss += args["l2"]*regularize_layer_params(l_out, l2)
        loss_det += args["l2"]*regularize_layer_params(l_out, l2)
    params = get_all_params(l_out, trainable=True)
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
    X_train, y_train, X_valid, y_valid = data
    #y_train = T.cast(y_train, "int32")
    #y_valid = T.cast(y_valid, "int32")
    train_fn = theano.function(inputs=[X,y], outputs=loss, updates=updates)
    loss_fn = theano.function(inputs=[X,y], outputs=loss_det)
    preds_fn = theano.function(inputs=[X], outputs=T.argmax(net_out_det,axis=1))
    
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "preds_fn": preds_fn,
        "l_out": l_out,
        "learning_rate": learning_rate,
        "bs": bs
    }


# In[26]:

def iterate(X_arr, y_arr, bs, augment):
    assert X_arr.shape[0] == y_arr.shape[0]
    b = 0
    while True:
        if b*bs >= X_arr.shape[0]:
            break
        this_X, this_y = X_arr[b*bs:(b+1)*bs], y_arr[b*bs:(b+1)*bs]
        # we need to do this for the training set
        # the valid/test sets are ok
        if augment:
            # ok, we must take a random crop that is 32x32
            new_this_X = []
            for i in range(0, this_X.shape[0]):
                rand_x, rand_y = np.random.randint(0,9), np.random.randint(0,9)
                new_this_X.append(this_X[i, :, rand_x : rand_x+32, rand_y : rand_y+32])
            new_this_X = np.asarray(new_this_X, dtype=this_X.dtype)
            yield new_this_X, this_y
        else:
            yield this_X, this_y
        # ---
        b += 1


# In[31]:

def train(net_cfg, 
          num_epochs,
          data,
          out_file=None,
          print_out=True,
          debug=False,
          resume=None,
          schedule={}):
    # prepare the out_file
    l_out = net_cfg["l_out"]
    f = None
    if resume == None:
        if out_file != None:
            f = open("%s.txt" % out_file, "wb")
            f.write("epoch,train_loss,avg_valid_loss,valid_accuracy,time\n")
        if print_out:
            print "epoch,train_loss,avg_valid_loss,valid_accuracy,time"
    else:
        sys.stderr.write("resuming training...\n")
        if out_file != None:
            f = open("%s.txt" % out_file, "ab")
        with open(resume) as g:
            set_all_param_values(l_out, pickle.load(g))          
    # extract functions
    X_train, y_train, X_valid, y_valid = data
    train_fn, loss_fn, preds_fn = net_cfg["train_fn"], net_cfg["loss_fn"], net_cfg["preds_fn"]
    learning_rate = net_cfg["learning_rate"]
    bs = net_cfg["bs"]
    
    # training
    train_idxs = [x for x in range(0, X_train.shape[0])]
    
    if debug:
        sys.stderr.write("idxs: %s\n" % train_idxs)
    for epoch in range(0, num_epochs):
        
        if epoch+1 in schedule:
            sys.stderr.write("changing learning rate to: %f" % schedule[epoch+1])
            learning_rate.set_value( floatX(schedule[epoch+1]) )
        
        np.random.shuffle(train_idxs)
        X_train = X_train[train_idxs]
        y_train = y_train[train_idxs]
        
        # training loop
        this_train_losses = []
        t0 = time()
        for X_train_batch, y_train_batch in iterate(X_train, y_train, bs, True):
            this_train_losses.append( train_fn(X_train_batch, y_train_batch) )
        time_taken = time() - t0
        
        # validation loss loop
        this_valid_losses = []
        for X_valid_batch, y_valid_batch in iterate(X_valid, y_valid, bs, False):
            this_valid_losses.append( loss_fn(X_valid_batch, y_valid_batch) )
        avg_valid_loss = np.mean(this_valid_losses)
        
        # validation accuracy loop
        this_valid_preds = []
        for X_valid_batch, _ in iterate(X_valid, y_valid, bs, False):
            this_valid_preds += preds_fn(X_valid_batch).tolist()
        valid_acc = np.mean( this_valid_preds == y_valid )
        
        ## ------------ ##
        if f != None:
            f.write(
                "%i,%f,%f,%f,%f\n" %
                    (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, time_taken) 
            )
            f.flush()
        if print_out:
            print "%i,%f,%f,%f,%f" %                 (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, time_taken)
        #print valid_loss
        #return train_losses
        
        with open("models/%s.model.%i" % (os.path.basename(out_file),epoch+1), "wb") as g:
            pickle.dump(get_all_param_values(l_out), g, pickle.HIGHEST_PROTOCOL) 
            
    if f != None:
        f.close()


# ----

# Reproduce stochastic depth paper using varying values of $p$ (ie no linear decay).

# In[2]:

if "CIFAR10_EXP_1" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0,1,2]:
        for p in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            #old experiments used this
            #np.random.seed(replicate)
            lasagne.random.set_rng(np.random.RandomState(replicate))
            this_args = {}
            out_file = "%s/p%f_stochastic_depth.%i" % (out_folder, p, replicate)
            if os.path.isfile("%s.txt" % out_file):
                continue
            train(
                get_net(
                    yu_cifar10_net({"survival_p": p, "nonlinearity_p": None}),
                    (X_train, y_train, X_valid, y_valid), 
                    {"batch_size": 128}
                ),
                num_epochs=20,
                data=(X_train, y_train, X_valid, y_valid),
                out_file=out_file,
                debug=False
            )


# Do stochastic nonlinearities.

# In[ ]:

if "CIFAR10_EXP_2" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0,1,2]:
        for p in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            lasagne.random.set_rng(np.random.RandomState(replicate))
            this_args = {}
            out_file = "%s/p%f_stochastic_nonlinearity.%i" % (out_folder, p, replicate)
            if os.path.isfile("%s.txt" % out_file):
                continue
            train(
                get_net(
                    yu_cifar10_net({"survival_p": None, "nonlinearity_p": p}),
                    (X_train, y_train, X_valid, y_valid), 
                    {"batch_size": 128}
                ),
                num_epochs=20,
                data=(X_train, y_train, X_valid, y_valid),
                out_file=out_file,
                debug=False
            )


# In[19]:

# linear decay schedule for stochastic depth
if "CIFAR10_EXP_3" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0,1,2]:
        #old experiments used this
        #np.random.seed(replicate)
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/stochastic_depth_decay0.5.%i" % (out_folder, replicate)
        if os.path.isfile("%s.txt" % out_file):
            continue
        train(
            get_net(
                yu_cifar10_net_decay({"decay":"depth"}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128}
            ),
            num_epochs=40,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False
        )


# In[26]:

# linear decay schedule for stochastic nonlinearity
if "CIFAR10_EXP_4" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0,1,2]:
        #old experiments used this
        #np.random.seed(replicate)
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/stochastic_nonlinearity_decay0.5.%i" % (out_folder, replicate)
        if os.path.isfile("%s.txt" % out_file):
            continue
        train(
            get_net(
                yu_cifar10_net_decay({"decay":"nonlinearity"}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128}
            ),
            num_epochs=40,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False
        )


# In[27]:

# linear decay schedule for stochastic nonlinearity
if "CIFAR10_EXP_5" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0,1,2]:
        #old experiments used this
        #np.random.seed(replicate)
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/stochastic_both_decay0.5.%i" % (out_folder, replicate)
        if os.path.isfile("%s.txt" % out_file):
            continue
        train(
            get_net(
                yu_cifar10_net_decay({"decay":"both"}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128}
            ),
            num_epochs=40,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False
        )


# ----

# In[23]:

# long experiment for stochastic depth
if "CIFAR10_EXP_7" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        if "AUGMENT" not in os.environ:
            out_file = "%s/long_stochastic_depth_decay0.5.%i" % (out_folder, replicate)
        else:
            out_file = "%s/long_augment_stochastic_depth_decay0.5.%i" % (out_folder, replicate)
        if os.path.isfile("%s.txt" % out_file):
            continue
        train(
            get_net(
                yu_cifar10_net_decay({"decay":"depth"}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128}
            ),
            num_epochs=200,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False
        )


# In[24]:

# long experiment for stochastic nonlinearity
if "CIFAR10_EXP_8" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        if "AUGMENT" not in os.environ:
            out_file = "%s/long_stochastic_nonlinearity_decay0.5.%i" % (out_folder, replicate)
        else:
            out_file = "%s/long_augment_stochastic_nonlinearity_decay0.5.%i" % (out_folder, replicate)
        if os.path.isfile("%s.txt" % out_file):
            continue
        train(
            get_net(
                yu_cifar10_net_decay({"decay":"nonlinearity"}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128}
            ),
            num_epochs=200,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False
        )


# In[27]:

# long experiment for stochastic depth using rmsprop... just curious
if "CIFAR10_EXP_7R" in os.environ and "AUGMENT" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        out_file = "%s/long_augment_rmsprop_stochastic_depth_decay0.5.%i" % (out_folder, replicate)
        if os.path.isfile("%s.txt" % out_file):
            continue
        train(
            get_net(
                yu_cifar10_net_decay({"decay":"depth"}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128, "rmsprop": True}
            ),
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False
        )


# In[28]:

# long experiment for stochastic nonlinearity... just curious
if "CIFAR10_EXP_8R" in os.environ and "AUGMENT" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_augment_rmsprop_stochastic_nonlinearity_decay0.5.%i" % (out_folder, replicate)
        if os.path.isfile("%s.txt" % out_file):
            continue
        train(
            get_net(
                yu_cifar10_net_decay({"decay":"nonlinearity"}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128, "rmsprop": True}
            ),
            num_epochs=1000,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False
        )


# -----

# We want to try and reproduce the results as best as possible from: https://github.com/yueatsprograms/Stochastic_Depth/blob/master/main.lua
# 
# Need to:
# 
# * Use He init
# * Use learning rate schedule
# * Do the random translation crop thing
# * Use an L2 weight decay of 1e-4

# In[36]:

if "LONG_BASELINE" in os.environ and "AUGMENT" in os.environ:  
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_baseline_basic_augment.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net({"init":"he", "survival_p":None, "nonlinearity_p": None}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128, "l2":1e-4}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={250: 0.001, 375: 0.0001}
        )
# 0.5, 0.75 for cifar10


##########################################
##########################################
##                                      ##
## EXPERIMENTS ON BASELINE RESNET       ##
##                                      ##
##########################################
##########################################

if "LONG_BASELINE_2" in os.environ:  
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_baseline_more_augment_lr0.1_leto18.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net({"init":"he", "survival_p":None, "nonlinearity_p": None}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={}
        )

if "LONG_BASELINE_2_RESUME" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_baseline_more_augment_lr0.1_leto18.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net({"init":"he", "survival_p":None, "nonlinearity_p": None}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.01}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            resume="models/long_baseline_more_augment_lr0.1_leto18.0.model.170"
        )

if "LONG_BASELINE_2_RESUME_2" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_baseline_more_augment_lr0.1_leto18.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net({"init":"he", "survival_p":None, "nonlinearity_p": None}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.001}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            resume="models/long_baseline_more_augment_lr0.1_leto18.0.model.2.220"
        )

if "LONG_BASELINE_2_REPLICATE" in os.environ:  
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [1]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_baseline_more_augment_lr0.1_leto04.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net({"init":"he", "survival_p":None, "nonlinearity_p": None}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={171: 0.01, 221: 0.001}
        )






##########################################
##########################################
##                                      ##
## EXPERIMENTS ON DEPTH RESNET          ##
##                                      ##
##########################################
##########################################


if "LONG_DEPTH" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_depth_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "depth"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={}
        )

if "LONG_DEPTH_RESUME" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_depth_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "depth"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.01}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={},
            resume="models/long_depth_more_augment_lr0.1.0.model.187.bak"
        )

if "LONG_DEPTH_RESUME_2" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_depth_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "depth"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.001}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={},
            resume="models/long_depth_more_augment_lr0.1.0.model.248.bak2"
        )

if "LONG_DEPTH_REPLICATE" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [1]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_depth_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "depth"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={188: 0.01, 249: 0.001}
        )



##########################################
##########################################
##                                      ##
## EXPERIMENTS ON NONLINEARITY RESNET   ##
##                                      ##
##########################################
##########################################

if "LONG_NONLINEARITY" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_nonlinearity_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "nonlinearity"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={}
        )

if "LONG_NONLINEARITY_RESUME" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_nonlinearity_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "nonlinearity"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.01}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={},
	        resume="models/long_nonlinearity_more_augment_lr0.1.0.model.167.bak"
        )


if "LONG_NONLINEARITY_RESUME_2" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_nonlinearity_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "nonlinearity"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.001}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={},
            resume="models/long_nonlinearity_more_augment_lr0.1.0.model.231.bak2"
        )

if "LONG_NONLINEARITY_REPLICATE" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [1]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_nonlinearity_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "nonlinearity"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={168: 0.01, 232: 0.001}
        )




if "LONG_BOTH" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_both_more_augment_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "both"}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={},
            resume="models/long_both_more_augment_lr0.1.0.model.500.bak"
        )


##########################################
##########################################
##                                      ##
## EXPERIMENTS ON ELU RESNET            ##
##                                      ##
##########################################
##########################################

if "LONG_BASELINE_ELU" in os.environ:  
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_baseline_more_augment_elu_lr0.1_leto18.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net({"init":"he", "survival_p":None, "nonlinearity_p": None, "nonlinearity": elu}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={}
        )

if "LONG_BASELINE_ELU_RESUME" in os.environ:  
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_baseline_more_augment_elu_lr0.1_leto18.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net({"init":"he", "survival_p":None, "nonlinearity_p": None, "nonlinearity": elu}),
                (X_train, y_train, X_valid, y_valid), 
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.01}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={},
            resume="models/long_baseline_more_augment_elu_lr0.1_leto18.0.model.204.bak"
        )

if "LONG_NONLINEARITY_ELU" in os.environ:
    out_folder = "output_stochastic_depth_resnet_new"
    for replicate in [0]:
        lasagne.random.set_rng(np.random.RandomState(replicate))
        this_args = {}
        out_file = "%s/long_nonlinearity_more_augment_elu_lr0.1.%i" % (out_folder, replicate)
        train(
            get_net(
                yu_cifar10_net_decay({"init":"he", "decay": "nonlinearity", "nonlinearity": elu}),
                (X_train, y_train, X_valid, y_valid),
                {"batch_size": 128, "l2":1e-4, "learning_rate":0.1}
            ),
            num_epochs=500,
            data=(X_train, y_train, X_valid, y_valid),
            out_file=out_file,
            debug=False,
            schedule={}
        )

# -----

# ## Plot the curves for stochastic depth

# In[11]:

get_ipython().run_cell_magic(u'R', u'', u'source("helper.R")\n\nps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "_stochastic_depth.", \n    "train_loss", \n    "stochastic depth", \n    "topright"\n)')


# Plot the validation curves (non-averaging)

# In[12]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "_stochastic_depth.", \n    "valid_accuracy", \n    "stochastic depth", \n    "bottomright"\n)\n')


# In[14]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\n#par(mfrow=c(2,2))\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "_stochastic_depth.", \n    "valid_accuracy", \n    "stochastic depth", \n    "bottomright",\n    TRUE\n)')


# In[29]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\npar(mfrow=c(2,3))\nout_folder = "output_stochastic_depth_resnet_new/"\n\nfor(i in 1:length(ps)) {\n    df = read.csv( paste(out_folder, ps[i], "_stochastic_depth.0.txt",sep="") )\n    plot(df$avg_valid_loss, type="l", col="red",\n         ylim=c(0,5), xlab="epoch", ylab="train/valid loss", main=ps[i])\n    lines(df$train_loss, col="blue")\n}')


# ## Plot the curves for stochastic nonlinearity

# In[15]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "_stochastic_nonlinearity.", \n    "train_loss", \n    "stochastic nonlinearity", \n    "topright"\n)')


# In[17]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "_stochastic_nonlinearity.", \n    "valid_accuracy", \n    "stochastic nonlinearity", \n    "bottomright"\n)\n\n')


# In[18]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "_stochastic_nonlinearity.", \n    "valid_accuracy", \n    "stochastic nonlinearity", \n    "bottomright",\n    TRUE\n)')


# In[30]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\n\npar(mfrow=c(2,3))\nout_folder = "output_stochastic_depth_resnet_new/"\n\nfor(i in 1:length(ps)) {\n    df = read.csv( paste(out_folder, ps[i], "_stochastic_nonlinearity.0.txt",sep="") )\n    plot(df$avg_valid_loss, type="l", col="red",\n         ylim=c(0,5), xlab="epoch", ylab="train/valid loss", main=ps[i])\n    lines(df$train_loss, col="blue")\n}')


# ## Plot the curves for stochastic depth with linear decay schedule

# In[22]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'\nps = c("")\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "stochastic_depth_decay0.5.", \n    "train_loss", \n    "stochastic depth", \n    "bottomright"\n)')


# In[23]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'\nps = c("")\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "stochastic_depth_decay0.5.", \n    "valid_accuracy", \n    "stochastic depth", \n    "bottomright"\n)')


# ## Plot the curves for stochastic nonlinearity with linear decay schedule

# In[24]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'\nps = c("")\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "stochastic_nonlinearity_decay0.5.", \n    "train_loss", \n    "stochastic nonlinearity", \n    "bottomright"\n)')


# In[25]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'\nps = c("")\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "stochastic_nonlinearity_decay0.5.", \n    "valid_accuracy", \n    "stochastic nonlinearity", \n    "bottomright"\n)')


# ## Plot the curves for stochastic depth + nonlinearity with linear decay schedule

# In[26]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'\nps = c("")\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "stochastic_both_decay0.5.", \n    "train_loss", \n    "stochastic both", \n    "bottomright"\n)')


# In[27]:

get_ipython().run_cell_magic(u'R', u'-w 480 -h 480', u'\nps = c("")\n\nplot_results(\n    ps, \n    "output_stochastic_depth_resnet_new/", \n    "stochastic_both_decay0.5.", \n    "valid_accuracy", \n    "stochastic both", \n    "bottomright"\n)')


# ----

# In[31]:

get_ipython().run_cell_magic(u'R', u'', u'ps = c(\n    "p1.000000",\n    "p0.900000",\n    "p0.800000",\n    "p0.700000",\n    "p0.600000",\n    "p0.500000"\n)\nfor(i in 1:length(ps)) {\n    df_nonlinearity = read.csv(paste("output_stochastic_depth_resnet_new/",ps[i],"_stochastic_nonlinearity.",0,".txt",sep=""))\n    df_nonlinearity2 = read.csv(paste("output_stochastic_depth_resnet_new/",ps[i],"_stochastic_nonlinearity.",1,".txt",sep=""))\n    df_nonlinearity3 = read.csv(paste("output_stochastic_depth_resnet_new/",ps[i],"_stochastic_nonlinearity.",2,".txt",sep=""))\n\n    df_stochastic = read.csv(paste("output_stochastic_depth_resnet_new/",ps[i],"_stochastic_depth.",0,".txt",sep=""))\n    df_stochastic2 = read.csv(paste("output_stochastic_depth_resnet_new/",ps[i],"_stochastic_depth.",1,".txt",sep=""))\n    df_stochastic3 = read.csv(paste("output_stochastic_depth_resnet_new/",ps[i],"_stochastic_depth.",2,".txt",sep=""))\n    \n    # valid loss between both\n    plot(df_nonlinearity$valid_accuracy, ylim=c(0,1), type="l", col="blue",\n        ylab="valid accuracy", xlab="epoch", main=paste("stochastic depth vs nonlinearity", ps[i]))\n    lines(df_nonlinearity2$valid_accuracy, col="blue")\n    lines(df_nonlinearity3$valid_accuracy, col="blue")\n    \n    lines(df_stochastic$valid_accuracy, col="red")\n    lines(df_stochastic2$valid_accuracy, col="red")\n    lines(df_stochastic3$valid_accuracy, col="red")\n    \n    legend("bottomright", legend=c("nonlinearity", "depth"), fill=c("blue", "red"))\n    \n}')


# ----

# In[75]:

import draw_net
reload(draw_net)


# In[71]:

tmp_net = yu_cifar10_net({"survival_p": 0.5, "nonlinearity_p": 0.0})


# In[79]:

draw_net.draw_to_file(get_all_layers(tmp_net), "network_diagram_stochastic_depth.png")


# In[82]:

tmp_net = yu_cifar10_net({"survival_p": 1.0, "nonlinearity_p": 0.5})


# In[83]:

draw_net.draw_to_file(get_all_layers(tmp_net), "network_diagram_stochastic_nonlinearity.png")


# ---

# In[22]:

lasagne.random.set_rng(np.random.RandomState(1))
l_in = InputLayer( (None, 10) )
l_dense = DenseLayer(l_in, num_units=5)
tmp = theano.function([l_in.input_var], get_output(l_dense, l_in.input_var))


# In[23]:

tmp( np.ones((1,10)) )


# In[ ]:



