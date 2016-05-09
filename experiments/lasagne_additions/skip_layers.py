
# coding: utf-8

# In[1]:

import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.random import get_rng
from lasagne.updates import *
import numpy as np
import sys
sys.path.append("../../modules/")
import helper as hp

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import os
import cPickle as pickle

from theano.tensor import TensorType

from theano.ifelse import ifelse

from time import time

get_ipython().magic(u'load_ext rpy2.ipython')

from scipy import stats


# In[192]:

"""
not working yet
"""
class SkipLayer(Layer):
    def __init__(self, incoming, wrapped_layer, p=0.5, **kwargs):
        super(SkipLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.wrapped_layer = wrapped_layer
        self.incoming = incoming
        self.p = p
    def get_output_for(self, input, deterministic=False, **kwargs):
        # if deterministic, use the underlying layer
        if deterministic or self.p == 0:
            return self.wrapped_layer.get_output_for(input, deterministic=False, **kwargs)
        else:
            # if uniform(0,1) < p, skip the underlying layer
            u = np.random.normal(0, 1)
            if u < self.p:
                print "ignoring"
                return input
            else:
                return self.wrapped_layer.get_output_for(input, deterministic=False, **kwargs)


# ----

# In[3]:

"""
rs = np.random.RandomState(1234)
rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))
shp = (10,8,26,26)
#new_shp = np.asarray([shp[0]]).reshape(shp.shape)

mask = rng.binomial( n=1, p=(0.5), size=(10,1,1,1) )
print mask.broadcastable
mask = T.addbroadcast(mask, *[x for x in range(1, len(shp))])
print mask.broadcastable
print (mask * T.ones((10,8,26,26))).eval()
"""


# In[2]:

class SkippableNonlinearityLayer(Layer):
    def __init__(self, incoming, nonlinearity=rectify, p=0.5, max_=10,
                 **kwargs):
        super(SkippableNonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.max_ = max_

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic and self.p == 0.0:
            # apply the bernoulli expectation
            return self.p*input + (1-self.p)*self.nonlinearity(input)
        else:
            # if uniform(0,1) < p, then apply I(x), else g(x)
            # so the probability of applying a nonlinearity
            # is 1-p
            """
            return ifelse(
                T.lt(self._srng.uniform( (1,), 0, 1)[0], self.p),
                input,
                self.nonlinearity(input)
            )
            """
            #print input.ndim
            
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
            
            #return input
            


# In[89]:

shallow_net = [
    ("conv", 3, 16, 1),
    ("maxpool", 2),
    ("conv", 3, 32, 1),
    ("maxpool", 2),
    ("conv", 3, 64, 1),
    ("dense", 128)
]


# In[90]:

deep_net = [
    ("conv", 3, 16, 1),
    ("conv", 3, 16, 1),
    ("conv", 3, 16, 1),
    
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    
    ("conv", 3, 64, 1),
    ("conv", 3, 64, 1),
    ("conv", 3, 64, 1),
    ("conv", 3, 64, 1),
    
    ("dense", 128) 
]


# In[50]:

def get_deep_net_light(args):
    l_in = InputLayer( (None, 1, 28, 28))
    l_prev = l_in
    for i in range(0, 3):
        l_prev = SkippableNonlinearityLayer(
            Conv2DLayer(l_prev, num_filters=8, filter_size=3, stride=1, nonlinearity=linear),
            nonlinearity=args["nonlinearity"],
            p=args["p"]
        )
    for i in range(0, 6):
        l_prev = SkippableNonlinearityLayer(
            Conv2DLayer(l_prev, num_filters=16, filter_size=3, stride=1, nonlinearity=linear),
            nonlinearity=args["nonlinearity"],
            p=args["p"]
        )
    for i in range(0, 4):
        l_prev = SkippableNonlinearityLayer(
            Conv2DLayer(l_prev, num_filters=32, filter_size=3, stride=1, nonlinearity=linear),
            nonlinearity=args["nonlinearity"],
            p=args["p"]
        )
    l_dense = SkippableNonlinearityLayer(
        DenseLayer(l_prev, num_units=128, nonlinearity=linear),
        nonlinearity=args["nonlinearity"],
        p=args["p"]
    )                                    
    l_out = DenseLayer(l_dense, num_units=10, nonlinearity=softmax)
    for layer in get_all_layers(l_out):
        if isinstance(layer, SkippableNonlinearityLayer):
            continue
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    sys.stderr.write(str(count_params(l_out)) + "\n")
    return l_out


# In[49]:

def get_deep_net(args):
    l_in = InputLayer( (None, 1, 28, 28))
    l_prev = l_in
    for i in range(0, 3):
        l_prev = SkippableNonlinearityLayer(
            Conv2DLayer(l_prev, num_filters=16, filter_size=3, stride=1, nonlinearity=linear),
            nonlinearity=args["nonlinearity"],
            p=args["p"]
        )
    for i in range(0, 6):
        l_prev = SkippableNonlinearityLayer(
            Conv2DLayer(l_prev, num_filters=32, filter_size=3, stride=1, nonlinearity=linear),
            nonlinearity=args["nonlinearity"],
            p=args["p"]
        )
    for i in range(0, 4):
        l_prev = SkippableNonlinearityLayer(
            Conv2DLayer(l_prev, num_filters=64, filter_size=3, stride=1, nonlinearity=linear),
            nonlinearity=args["nonlinearity"],
            p=args["p"]
        )
    l_dense = SkippableNonlinearityLayer(
        DenseLayer(l_prev, num_units=128, nonlinearity=linear),
        nonlinearity=args["nonlinearity"],
        p=args["p"]
    )                                    
    l_out = DenseLayer(l_dense, num_units=10, nonlinearity=softmax)
    for layer in get_all_layers(l_out):
        if isinstance(layer, SkippableNonlinearityLayer):
            continue
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    sys.stderr.write(str(count_params(l_out)) + "\n")
    return l_out


# In[46]:

def get_basic_net(args):
    l_in = InputLayer( (None, 1, 28, 28))
    l_prev = l_in
    for i in range(0, 1):
        l_prev = Conv2DLayer(l_prev, num_filters=8, filter_size=3, stride=1, nonlinearity=args["nonlinearity"])
        l_prev = MaxPool2DLayer(l_prev, pool_size=2)
    for i in range(0, 1):
        l_prev = Conv2DLayer(l_prev, num_filters=16, filter_size=3, stride=1, nonlinearity=args["nonlinearity"])
        l_prev = MaxPool2DLayer(l_prev, pool_size=2)
    for i in range(0, 1):
        l_prev = Conv2DLayer(l_prev, num_filters=32, filter_size=3, stride=1, nonlinearity=args["nonlinearity"])
        l_prev = MaxPool2DLayer(l_prev, pool_size=2)
    l_dense = DenseLayer(l_prev, num_units=128, nonlinearity=args["nonlinearity"])
    
    l_out = DenseLayer(l_dense, num_units=10, nonlinearity=softmax)
    for layer in get_all_layers(l_out):
        if isinstance(layer, SkippableNonlinearityLayer):
            continue
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    sys.stderr.write(str(count_params(l_out)) + "\n")
    return l_out


# In[29]:

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
    params = get_all_params(l_out, trainable=True)
    if "max_norm" in args:
        grads = total_norm_constraint( T.grad(loss, params), max_norm=args["max_norm"])
    else:
        grads = T.grad(loss, params)
    updates = nesterov_momentum(grads, params, learning_rate=0.01, momentum=0.9)
    # index fns
    bs = args["batch_size"]
    X_train, y_train, X_valid, y_valid = data
    y_train = T.cast(y_train, "int32")
    y_valid = T.cast(y_valid, "int32")
    train_fn = theano.function(inputs=[idx], outputs=loss, updates=updates, 
        givens={X: X_train[idx*bs : (idx+1)*bs], y: y_train[idx*bs : (idx+1)*bs]}
    )
    # this is for the validation set
    # make the output deterministic
    loss_fn = theano.function(inputs=[], outputs=loss_det, givens={X: X_valid,y: y_valid})
    acc = T.mean( T.eq( T.argmax(net_out_det,axis=1), y_valid) )
    acc_fn = theano.function(inputs=[], outputs=acc, givens={X:X_valid})
    # this is meant to be non-deterministic
    preds = T.argmax(net_out,axis=1)
    preds_fn = theano.function(inputs=[], outputs=preds, givens={X:X_valid})
    # this is also meant to be non-deterministic
    out_fn = theano.function(inputs=[], outputs=net_out, givens={X:X_valid})

    tmp_fn = theano.function([X], net_out)
    outs_with_nonlinearity = theano.function(
        [X], [ get_output(layer, X, deterministic=True) for layer in get_all_layers(l_out) 
              if isinstance(layer, SkippableNonlinearityLayer) ], on_unused_input="warn"
    )
    outs_without_nonlinearity = theano.function(
        [X], [ get_output(layer, X, deterministic=True) for layer in get_all_layers(l_out) 
              if isinstance(layer, Conv2DLayer) or isinstance(layer, DenseLayer) ]
    )
    return {
        "train_fn": train_fn,
        "acc_fn": acc_fn,
        "preds_fn": preds_fn,
        "loss_fn": loss_fn,
        "out_fn": out_fn,
        "outs_with_nonlinearity": outs_with_nonlinearity,
        "outs_without_nonlinearity": outs_without_nonlinearity,
        "l_out": l_out,
        "tmp_fn": tmp_fn,
        "n_batches": X_train.get_value().shape[0] // bs
    }


# In[10]:

train_data, valid_data, _ = hp.load_mnist("../../data/mnist.pkl.gz")
X_train, y_train = train_data
X_valid, y_valid = valid_data
# minimal
X_train_minimal = X_train[0:200]
y_train_minimal = y_train[0:200]
# ---
X_train = theano.shared(np.asarray(X_train, dtype=theano.config.floatX), borrow=True)
y_train = theano.shared(np.asarray(y_train, dtype=theano.config.floatX), borrow=True)
X_valid = theano.shared(np.asarray(X_valid, dtype=theano.config.floatX), borrow=True)
y_valid = theano.shared(np.asarray(y_valid, dtype=theano.config.floatX), borrow=True)
# minimal
X_train_minimal = theano.shared(np.asarray(X_train_minimal, dtype=theano.config.floatX), borrow=True)
y_train_minimal = theano.shared(np.asarray(y_train_minimal, dtype=theano.config.floatX), borrow=True)
# ---
#y_train = T.cast(y_train, "int32")
#y_valid = T.cast(y_valid, "int32")
# minimal
#y_train_minimal = T.cast(y_train_minimal, "int32")
# ---


# In[22]:

"""
DON'T USE
"""
def iterate(X_train, y_train, batch_size):
    b = 0
    idxs = [x for x in range(0, X_train.shape[0])]
    while True:
        np.random.shuffle(idxs)
        if b*batch_size >= X_train.shape[0]:
            break
        X_batch = X_train[idxs][b*batch_size:(b+1)*batch_size]
        y_batch = y_train[idxs][b*batch_size:(b+1)*batch_size] 
        b += 1
        yield X_batch, y_batch


# In[19]:

def train(net_cfg, 
          num_epochs,
          data,
          out_file=None,
          print_out=True,
          debug=False):
    # prepare the out_file
    f = None
    if out_file != None:
        f = open("%s.txt" % out_file, "wb")
        f.write("epoch,train_loss,valid_loss,valid_accuracy,valid_accuracy_ensemble,time\n")
    if print_out:
        print "epoch,train_loss,valid_loss,valid_accuracy,valid_accuracy_ensemble,time"
    # extract functions
    X_train, y_train, X_valid, y_valid = data
    train_fn = net_cfg["train_fn"]
    loss_fn = net_cfg["loss_fn"]
    acc_fn = net_cfg["acc_fn"]
    preds_fn = net_cfg["preds_fn"]
    outs_with_nonlinearity = net_cfg["outs_with_nonlinearity"]
    # training
    n_batches = net_cfg["n_batches"]
    idxs = [x for x in range(0, n_batches)]
    if debug:
        sys.stderr.write("n_batches: %s\n" % n_batches)
        sys.stderr.write("idxs: %s\n" % idxs)
    for epoch in range(0, num_epochs):
        this_train_losses = []
        np.random.shuffle(idxs)
        #if debug:
        #    sys.stderr.write("%s\n" % idxs)
        t0 = time()
        for i in range(0, len(idxs)):
            loss_for_this_batch = train_fn( idxs[i] )
            this_train_losses.append( loss_for_this_batch )
        time_taken = time() - t0
        valid_loss = loss_fn()
        valid_acc = acc_fn()
        ## EXPERIMENTAL ##
        mat = []
        numiters=10
        for i in range(0, numiters):
            mat.append( preds_fn() )
        mat = stats.mode(np.vstack(mat))[0]
        valid_acc_ensemble = np.mean(mat==y_valid.get_value())
        ## ------------ ##
        if f != None:
            f.write(
                "%i,%f,%f,%f,%f,%f\n" %
                    (epoch+1, np.mean(this_train_losses), valid_loss, valid_acc, valid_acc_ensemble, time_taken) 
            )
        if print_out:
            print "%i,%f,%f,%f,%f,%f" %                 (epoch+1, np.mean(this_train_losses), valid_loss, valid_acc, valid_acc_ensemble, time_taken)
        #print valid_loss
        #return train_losses
    if f != None:
        f.close()
        # ok, now save the model as a pkl
        l_out = net_cfg["l_out"]
        with open("%s.model" % out_file, "wb") as g:
            pickle.dump(get_all_param_values(l_out), g, pickle.HIGHEST_PROTOCOL)


# If we train two networks: one with $p = 0$ and $p = 0.5$, we expect the latter to have activations that are not close to the saturation regime. This is because if $x$ is very big, $tanh(x)$ will be in the saturation regime, but when we compute $I(x) = x$, this will be massive, therefore significantly influencing the subsequent layers and forcing backprop to lower the magnitude of $x$.

# Let's verify the implementation using a dummy example.

# In[12]:

def test_this():
    l_in = InputLayer( (None, 5) )
    l_dense = DenseLayer(l_in, num_units=5, nonlinearity=linear, W=np.eye(5))
    l_id = SkippableNonlinearityLayer(l_dense, nonlinearity=sigmoid)
    X = T.fmatrix()
    out = get_output(l_id, X)
    return theano.function([X], out)
dummy_net_eval = test_this()


# In[17]:

dummy_net_eval( np.ones((4, 5), dtype="float32") )


# Let's try a "deep" net on MNIST, and see what the outputs look like, as a dummy example.

# In[53]:

dummy_net = get_net(
    l_out=get_deep_net_light({"p": 0.0, "nonlinearity": tanh}), 
    data=(X_train_minimal, y_train_minimal, X_train_minimal, y_train_minimal),
    args={"batch_size": 10, "max_norm": 10}
)
train(
    net_cfg=dummy_net,
    num_epochs=10,
    data=(X_train_minimal, y_train_minimal, X_train_minimal, y_train_minimal),
)


# ----

# In[16]:

skip_check = True


# In[17]:

if skip_check or os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
    for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
        for p in [0.0, 0.1, 0.25, 0.5, 0.75]:
            np.random.seed(0)
            train(
                get_net(
                    get_deep_net_light({"p":p, "nonlinearity": nonlinearity[1]}),
                    (X_train, y_train, X_valid, y_valid), 
                    {"batch_size": 128, "max_norm": 10}
                ),
                num_epochs=10,
                data=(X_train, y_train, X_valid, y_valid),
                out_file="output/p%f_%s" % (p, nonlinearity[0]),
                debug=False
            )


# Caveats about prelim exps:
# * Loss calculation on valid set not deterministic
# * Using lightweight deep network
# * --NaNs due to not using grad clipping--
# 
# * We would expect the stochastic nonlinearity to take relatively longer to train, because we are training an ensemble of networks (analogous to dropout)

# In[50]:

plt.plot(train_losses_without_id, "b-", train_losses_with_id, "r-")
plt.xlabel("# epochs")
plt.ylabel("train loss")


# -----

# Ok, let's look at all the tanh models

# In[134]:

deep_net_light = [
    ("conv", 3, 8, 1),
    ("conv", 3, 8, 1),
    ("conv", 3, 8, 1),
    
    ("conv", 3, 16, 1),
    ("conv", 3, 16, 1),
    ("conv", 3, 16, 1),
    ("conv", 3, 16, 1),
    ("conv", 3, 16, 1),
    ("conv", 3, 16, 1),
    
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    ("conv", 3, 32, 1),
    
    ("dense", 128) 
]


# In[255]:

models = dict()
for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
    models[nonlinearity[0]] = dict()
    for p in [0.0, 0.25, 0.5, 0.75]:
        with open("output/p%f_%s.model" % (p, nonlinearity[0])) as f:
            model = pickle.load(f)
        tmp_net = get_net(
            deep_net_light,
            (X_train, y_train, X_valid, y_valid),
            {"p":0.0, "nonlinearity": nonlinearity[1], "batch_size": 32}
        )
        models[nonlinearity[0]][p] = tmp_net
        set_all_param_values(models[nonlinearity[0]][p]["l_out"], model)


# Compare the tanh models with the baseline

# In[272]:

get_ipython().run_cell_magic(u'R', u'', u'dfb = read.csv("output/p0.000000_tanh.txt")\ndf25 = read.csv("output/p0.250000_tanh.txt")\ndf50 = read.csv("output/p0.500000_tanh.txt")\ndf75 = read.csv("output/p0.750000_tanh.txt")\npar(mfrow=c(1,2))\n# valid plots\nplot(dfb$train_loss, type="l", xlab="# epochs", ylab="valid loss", col="blue", ylim=c(0,1))\nlines(df25$train_loss, col="red")\nlines(df50$train_loss, col="orange")\nlines(df75$train_loss, col="green")\n# train plots\nplot(dfb$epoch, type="l", xlab="# epochs", ylab="train loss", col="blue", ylim=c(0,1))\nlines(df25$epoch, col="red")\nlines(df50$epoch, col="orange")\nlines(df75$epoch, col="green")\n')


# In[269]:

get_ipython().run_cell_magic(u'R', u'', u'dfb = read.csv("output/p0.000000_relu.txt")\ndf25 = read.csv("output/p0.250000_relu.txt")\ndf50 = read.csv("output/p0.500000_relu.txt")\ndf75 = read.csv("output/p0.750000_relu.txt")\npar(mfrow=c(1,2))\n# valid plots\nplot(dfb$train_loss, type="l", xlab="# epochs", ylab="valid loss", col="blue", ylim=c(0,0.5))\nlines(df25$train_loss, col="red")\nlines(df50$train_loss, col="orange")\nlines(df75$train_loss, col="green")\n# train plots\nplot(dfb$epoch, type="l", xlab="# epochs", ylab="train loss", col="blue", ylim=c(0,0.5))\nlines(df25$epoch, col="red")\nlines(df50$epoch, col="orange")\nlines(df75$epoch, col="green")\n')


# In[256]:

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8,6))
fig.tight_layout()
for i in range(0,4*3):
    plt.subplot(4,3,i)
    plt.figure
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.ylabel("g(x)")
    plt.xlabel("x")
    plt.plot(
        models["relu"][0.0]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        models["relu"][0.0]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        "bo",
        models["relu"][0.25]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        models["relu"][0.25]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        "ro",
    )


# In[207]:

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8,6))
fig.tight_layout()
for i in range(0,4*3):
    plt.subplot(4,3,i)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.ylabel("g(x)")
    plt.xlabel("x")
    plt.plot(
        models["relu"][0.0]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        models["relu"][0.0]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        "bo",
        models["relu"][0.5]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        models["relu"][0.5]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        "ro",        
    )


# In[209]:

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8,6))
fig.tight_layout()
for i in range(0,4*3):
    plt.subplot(4,3,i)
    plt.figure
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.ylabel("g(x)")
    plt.xlabel("x")
    plt.plot(
        models["relu"][0.0]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        models["relu"][0.0]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        "bo",
        models["relu"][0.75]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        models["relu"][0.75]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
        "ro",
    )


# In[ ]:




# ----

# In[181]:

plt.figure(figsize=(8,6))
for i in range(0,4):
    plt.subplot(2,2,i)
    plt.xlim(-4, 4)
    plt.ylim(-2, 2)
    plt.plot(
        no_id_net["outs_without_nonlinearity"](X_train[0:100])[i].flatten(),
        no_id_net["outs_with_nonlinearity"](X_train[0:100])[i].flatten(),
        "bo",
        id_net["outs_without_nonlinearity"](X_train[0:100])[i].flatten(),
        id_net["outs_with_nonlinearity"](X_train[0:100])[i].flatten(),
        "ro",
    )
    plt.ylabel("g(x)")
    plt.xlabel("x")


# In[49]:

get_ipython().magic(u'Rpush train_losses')


# In[51]:

train_losses_2 =     train(X_train, y_train, X_valid, y_valid, get_net({"p":0}), num_epochs=7, bs=32)


# In[52]:

get_ipython().magic(u'Rpush train_losses_2')


# In[59]:

get_ipython().run_cell_magic(u'R', u'', u'plot(train_losses_2, type="l", col="red")\nlines(train_losses, col="blue")')


# In[ ]:



