
# coding: utf-8

# In[114]:

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

import deep_residual_learning_CIFAR10


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


# In[110]:

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
        if deterministic or self.p == 0.0:
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
            


# In[111]:

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


# In[51]:

def get_deep_net_light(args, custom_layer=SkippableNonlinearityLayer):
    if "dropout" in args:
        sys.stderr.write("using dropout instead of skippable nonlinearity...\n")
    l_in = InputLayer( (None, 1, 28, 28))
    l_prev = l_in
    for i in range(0, 3):
        l_prev = Conv2DLayer(l_prev, num_filters=8, filter_size=3, stride=1, nonlinearity=linear)
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
    for i in range(0, 6):
        l_prev = Conv2DLayer(l_prev, num_filters=16, filter_size=3, stride=1, nonlinearity=linear)
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
    for i in range(0, 4):
        l_prev = Conv2DLayer(l_prev, num_filters=32, filter_size=3, stride=1, nonlinearity=linear)
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
            
    #l_dense = DenseLayer(l_prev, num_units=128, nonlinearity=linear)       
    if "dropout" not in args:
        l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
    else:
        l_prev = DropoutLayer( NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"]), p=args["p"] )
    
    l_out = DenseLayer(l_prev, num_units=10, nonlinearity=softmax) # in used to be l_dense
    for layer in get_all_layers(l_out):
        if isinstance(layer, custom_layer) or isinstance(layer, NonlinearityLayer):
            continue
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    sys.stderr.write(str(count_params(l_out)) + "\n")
    return l_out


# In[10]:

def get_deep_net_light_with_dense(args, custom_layer=SkippableNonlinearityLayer):
    if "dropout" in args:
        sys.stderr.write("using dropout instead of skippable nonlinearity...\n")
    l_in = InputLayer( (None, 1, 28, 28))
    l_prev = l_in
    for i in range(0, 3):
        l_prev = Conv2DLayer(l_prev, num_filters=8, filter_size=3, stride=1, nonlinearity=linear)
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
        else:
            l_prev = NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"])
    for i in range(0, 6):
        l_prev = Conv2DLayer(l_prev, num_filters=16, filter_size=3, stride=1, nonlinearity=linear)
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
        else:
            l_prev = NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"])
    for i in range(0, 4):
        l_prev = Conv2DLayer(l_prev, num_filters=32, filter_size=3, stride=1, nonlinearity=linear)
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
        else:
            l_prev = NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"])
            
    l_prev = DenseLayer(l_prev, num_units=64, nonlinearity=linear)       
    if "dropout" not in args:
        l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
    else:
        l_prev = DropoutLayer( NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"]), p=args["p"] )
    
    l_out = DenseLayer(l_prev, num_units=10, nonlinearity=softmax) # in used to be l_dense
    for layer in get_all_layers(l_out):
        #if isinstance(layer, custom_layer) or isinstance(layer, NonlinearityLayer):
        #    continue
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    sys.stderr.write(str(count_params(l_out)) + "\n")
    return l_out


# In[116]:

def get_deep_net_light_with_dense_for_cifar10(args, custom_layer=SkippableNonlinearityLayer):
    # f1, f2, f3, f4 = 8,16,32,64 for cifar_exp_1
    # f1, f2, f3, f4 = 32,64,128,256 for cifar_exp_1
    #
    if "dropout" in args:
        sys.stderr.write("using dropout instead of skippable nonlinearity...\n")
    if args["nonlinearity"] == rectify:
        gain = "relu"
        sys.stderr.write("get_deep_net_light_with_dense_for_cifar10(): relu gain...\n")
    else:
        gain = 1.0
    l_in = InputLayer( (None, 3, 32, 32))
    l_prev = l_in
    for i in range(0, 3):
        l_prev = Conv2DLayer(l_prev, num_filters=args["f1"], filter_size=3, stride=1, nonlinearity=linear, W=GlorotUniform(gain=gain))
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
        else:
            l_prev = NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"])
    for i in range(0, 5):
        l_prev = Conv2DLayer(l_prev, num_filters=args["f2"], filter_size=3, stride=1, nonlinearity=linear, W=GlorotUniform(gain=gain))
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
        else:
            l_prev = NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"])
    for i in range(0, 7):
        l_prev = Conv2DLayer(l_prev, num_filters=args["f3"], filter_size=3, stride=1, nonlinearity=linear, W=GlorotUniform(gain=gain))
        if "dropout" not in args:
            l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
        else:
            l_prev = NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"])
            
    l_prev = DenseLayer(l_prev, num_units=args["f4"], nonlinearity=linear, W=GlorotUniform(gain=gain))
    if "dropout" not in args:
        l_prev = custom_layer(l_prev, nonlinearity=args["nonlinearity"], p=args["p"])
    else:
        l_prev = DropoutLayer( NonlinearityLayer(l_prev, nonlinearity=args["nonlinearity"]), p=args["p"] )
    
    l_out = DenseLayer(l_prev, num_units=10, nonlinearity=softmax) # in used to be l_dense
    for layer in get_all_layers(l_out):
        #if isinstance(layer, custom_layer) or isinstance(layer, NonlinearityLayer):
        #    continue
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    sys.stderr.write(str(count_params(l_out)) + "\n")
    return l_out


# In[4]:

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


# In[ ]:




# In[5]:

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
    #l_dense = DenseLayer(l_prev, num_units=128, nonlinearity=args["nonlinearity"])
    
    l_out = DenseLayer(l_prev, num_units=10, nonlinearity=softmax)
    for layer in get_all_layers(l_out):
        if isinstance(layer, SkippableNonlinearityLayer):
            continue
        sys.stderr.write("%s,%s\n" % (layer, layer.output_shape))
    sys.stderr.write(str(count_params(l_out)) + "\n")
    return l_out


# In[113]:

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
        loss += args["l2"]*regularize_layer_params(l_out, l2)
        loss_det += args["l2"]*regularize_layer_params(l_out, l2)
    params = get_all_params(l_out, trainable=True)
    if "max_norm" in args:
        grads = total_norm_constraint( T.grad(loss, params), max_norm=args["max_norm"])
    else:
        grads = T.grad(loss, params)
    learning_rate = 0.01 if "learning_rate" not in args else args["learning_rate"]
    momentum = 0.9 if "momentum" not in args else args["momentum"]
    if "rmsprop" in args:
        sys.stderr.write("using rmsprop instead of nesterov momentum...\n")
        updates = rmsprop(grads, params, learning_rate=learning_rate)
    else:
        updates = nesterov_momentum(grads, params, learning_rate=learning_rate, momentum=momentum)
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
              if isinstance(layer, SkippableNonlinearityLayer) or \
                  isinstance(layer, MoreSkippableNonlinearityLayer) ], on_unused_input="warn"
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


# In[14]:

if "CIFAR10_EXP_1" not in os.environ:
    sys.stderr.write("loading mnist...\n")
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
else:
    sys.stderr.write("loading cifar10...\n")
    dat = deep_residual_learning_CIFAR10.load_data()
    X_train_and_valid = dat["X_train"]
    y_train_and_valid = dat["Y_train"]
    
    X_train_minimal = theano.shared(X_train_and_valid[0:100].astype(theano.config.floatX), borrow=True)
    y_train_minimal = theano.shared(y_train_and_valid[0:100].astype(theano.config.floatX), borrow=True)
    
    X_test = theano.shared(dat["X_test"].astype(theano.config.floatX), borrow=True)
    y_test = theano.shared(dat["Y_test"].astype(theano.config.floatX), borrow=True)
    n = X_train_and_valid.shape[0]
    X_train = theano.shared(X_train_and_valid[0 : 0.85*n].astype(theano.config.floatX), borrow=True)
    y_train = theano.shared(y_train_and_valid[0 : 0.85*n].astype(theano.config.floatX), borrow=True)
    X_valid = theano.shared(X_train_and_valid[0.85*n :: ].astype(theano.config.floatX), borrow=True)
    y_valid = theano.shared(y_train_and_valid[0.85*n :: ].astype(theano.config.floatX), borrow=True)
    


# In[34]:

"""
sys.stderr.write("debugging network architecture...\n")

get_net(
    l_out=get_deep_net_light_with_dense_for_cifar10(
        {"p": 0.25, "dropout": True, "nonlinearity": tanh},
        custom_layer=MoreSkippableNonlinearityLayer
    ), 
    data=(X_train_minimal, y_train_minimal, X_train_minimal, y_train_minimal),
    args={"batch_size": 10}
)
"""


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


# In[53]:

def train(net_cfg, 
          num_epochs,
          data,
          out_file=None,
          print_out=True,
          debug=False,
          resume=False):
    # prepare the out_file
    f = None
    if resume == False:
        if out_file != None:
            f = open("%s.txt" % out_file, "wb")
            f.write("epoch,train_loss,valid_loss,valid_accuracy,valid_accuracy_ensemble,time\n")
        if print_out:
            print "epoch,train_loss,valid_loss,valid_accuracy,valid_accuracy_ensemble,time"
    else:
        sys.stderr.write("resuming training...\n")
        if out_file != None:
            f = open("%s.txt" % out_file, "ab")
        l_out = net_cfg["l_out"]
        with open("%s.model" % out_file) as g:
            set_all_param_values(l_out, pickle.load(g))          
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

# In[34]:

def test_this():
    l_in = InputLayer( (None, 5) )
    # np.eye has dtype float64 default
    l_dense = DenseLayer(l_in, num_units=5, nonlinearity=linear, W=np.eye(5))
    l_id = SkippableNonlinearityLayer(l_dense, nonlinearity=sigmoid)
    X = T.fmatrix()
    out = get_output(l_id, X)
    return theano.function([X], out)
dummy_net_eval = test_this()


# In[17]:

dummy_net_eval( np.ones((4, 5), dtype="float32") )


# Let's try a "deep" net on MNIST, and see what the outputs look like, as a dummy example.

# In[137]:

#dummy_net = get_net(
#    l_out=get_deep_net_light_with_dense({"p": 0.25, "dropout": True, "nonlinearity": tanh}, custom_layer=MoreSkippableNonlinearityLayer), 
#    data=(X_train_minimal, y_train_minimal, X_train_minimal, y_train_minimal),
#    args={"batch_size": 10}
#)
#train(
#    net_cfg=dummy_net,
#    num_epochs=10,
#    data=(X_train_minimal, y_train_minimal, X_train_minimal, y_train_minimal),
#)


# ---

# In[27]:

skip_check = True


# In this experiment we vary $p$
# 
# CAVEAT: skippable nonlinearity was not added to the dense layer before the final

# In[121]:

"""
if skip_check or os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
    for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
        for p in [0.0, 0.1, 0.25, 0.5, 0.75]:
            np.random.seed(0)
            train(
                get_net(
                    get_deep_net_light({"p":p, "nonlinearity": nonlinearity[1]}),
                    (X_train, y_train, X_valid, y_valid), 
                    {"batch_size": 128, "max_norm": 2}
                ),
                num_epochs=10,
                data=(X_train, y_train, X_valid, y_valid),
                out_file="output/p%f_%s" % (p, nonlinearity[0]),
                debug=False
            )
"""


# In this experiment we experiment with $L_{2}$

# CAVEAT: GRADIENT CLIPPING WAS USED, AND MAX EPOCHS = 10

# In[129]:

"""
if skip_check or os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
    for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
        for lamb in [1e-2, 1e-3, 1e-4, 1e-5]:
            np.random.seed(0)
            train(
                get_net(
                    get_deep_net_light({"p":p, "nonlinearity": nonlinearity[1]}),
                    (X_train, y_train, X_valid, y_valid), 
                    {"batch_size": 128, "max_norm": 2, "l2": lamb}
                ),
                num_epochs=10,
                data=(X_train, y_train, X_valid, y_valid),
                out_file="output/p%f_%s_l%f" % (p, nonlinearity[0], lamb),
                debug=False
            )
"""


# In this experiment we play with $p$ but for dropout (do no gradient clipping in this one... we probably won't need it)

# In[15]:

"""
if skip_check or os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
    for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
        for p in [0.1, 0.25, 0.5, 0.75]:
            np.random.seed(0)
            train(
                get_net(
                    get_deep_net_light({"p":p, "dropout": True, "nonlinearity": nonlinearity[1]}),
                    (X_train, y_train, X_valid, y_valid), 
                    {"batch_size": 128}
                ),
                num_epochs=30,
                data=(X_train, y_train, X_valid, y_valid),
                out_file="output/p%f_%s_dropout" % (p, nonlinearity[0]),
                debug=False
            )
"""


# Do five replicates of tanh(0.5), relu(0.5), tanh_dropout(0.5), and relu_dropout(0.5)

# In[25]:

"""
if skip_check or os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
    for x in range(0, 5):
        for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
            for dropout in [True, False]:
                np.random.seed(x)
                print "output_replicate/p%f_%s_dropout%i.%i" % (0.5, nonlinearity[0], dropout, x) 
                get_net_args = {"batch_size": 128}
                if dropout == False:
                    get_net_args["max_norm"] = 2
                train(
                    get_net(
                        get_deep_net_light({"p":0.5, "dropout": dropout, "nonlinearity": nonlinearity[1]}),
                        (X_train, y_train, X_valid, y_valid), 
                        get_net_args
                    ),
                    num_epochs=30,
                    data=(X_train, y_train, X_valid, y_valid),
                    out_file="output_replicate/p%f_%s_dropout%i.%i" % (0.5, nonlinearity[0], dropout, x),
                    debug=False
                )
"""


# Caveats about prelim exps:
# * Using lightweight deep network
# 
# * We would expect the stochastic nonlinearity to take relatively longer to train, because we are training an ensemble of networks (analogous to dropout)

# ----

# In[52]:

if "MORE_SKIPPABLE_1" in os.environ:
    if skip_check or os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
        for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
            for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                np.random.seed(0)
                train(
                    get_net(
                        get_deep_net_light({"p":p, "nonlinearity": nonlinearity[1]}, custom_layer=MoreSkippableNonlinearityLayer),
                        (X_train, y_train, X_valid, y_valid), 
                        {"batch_size": 128}
                    ),
                    num_epochs=20,
                    data=(X_train, y_train, X_valid, y_valid),
                    out_file="output_more/p%f_%s" % (p, nonlinearity[0]),
                    debug=False
                )


# In[55]:

if "MORE_SKIPPABLE_2" in os.environ:
    if skip_check or os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
        for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
            for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
                np.random.seed(0)
                train(
                    get_net(
                        get_deep_net_light({"p":p, "dropout": True, "nonlinearity": nonlinearity[1]}, custom_layer=MoreSkippableNonlinearityLayer),
                        (X_train, y_train, X_valid, y_valid), 
                        {"batch_size": 128}
                    ),
                    num_epochs=20,
                    data=(X_train, y_train, X_valid, y_valid),
                    out_file="output_more/p%f_%s_dropout" % (p, nonlinearity[0]),
                    debug=False
                )


# start from here

# In[68]:

for replicate in [0, 1,2]:
    for dropout in [True, False]:
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
                if os.path.isfile("%s.txt" % out_file):
                    pass
                else:
                    print "cannot find: %s" % out_file


# In[63]:

if "MNIST_EXP_1" in os.environ:
    for replicate in [1,2]:
        for dropout in [True, False]:
            for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
                    np.random.seed(replicate)
                    this_args = {"p":p, "nonlinearity": nonlinearity[1]}
                    if dropout:
                        this_args["dropout"] = True
                        out_file="output_more/p%f_%s_with_dense_dropout.%i" % (p, nonlinearity[0], replicate)
                    else:                       
                        out_file="output_more/p%f_%s_with_dense.%i" % (p, nonlinearity[0], replicate)
                    if os.path.isfile("%s.txt" % out_file):
                        continue
                    train(
                        get_net(
                            get_deep_net_light_with_dense(this_args, custom_layer=MoreSkippableNonlinearityLayer),
                            (X_train, y_train, X_valid, y_valid), 
                            {"batch_size": 128}
                        ),
                        num_epochs=20,
                        data=(X_train, y_train, X_valid, y_valid),
                        out_file=out_file,
                        debug=False
                    )


# In[139]:

if "CIFAR10_EXP_1" in os.environ:
    #if "RMSPROP" not in os.environ:
    #    out_folder = "output_cifar10"
    #else:
    #    out_folder = "output_cifar10_rmsprop"
    out_folder = "output_cifar10"
    for replicate in [0,1,2]:
        for dropout in [True, False]:
            for p in [0.1, 0.2, 0.3, 0.4, 0.5]:              
                for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
                    np.random.seed(replicate)
                    this_args = {"p":p, "nonlinearity": nonlinearity[1], "f1":8, "f2":16, "f3": 32, "f4":64}
                    if dropout:
                        this_args["dropout"] = True
                        out_file = "%s/p%f_%s_with_dense_dropout.%i" % (out_folder, p, nonlinearity[0], replicate)
                    else:
                        out_file = "%s/p%f_%s_with_dense.%i" % (out_folder, p, nonlinearity[0], replicate)
                    #if "RMSPROP" in os.environ:
                    #    this_args["rmsrop"] = True
                    if os.path.isfile("%s.txt" % out_file):
                        continue
                    train(
                        get_net(
                            get_deep_net_light_with_dense_for_cifar10(
                                this_args, custom_layer=MoreSkippableNonlinearityLayer),
                            (X_train, y_train, X_valid, y_valid), 
                            {"batch_size": 128}
                        ),
                        num_epochs=20,
                        data=(X_train, y_train, X_valid, y_valid),
                        out_file=out_file,
                        debug=False
                    )


# In[119]:

if "CIFAR10_EXP_2" in os.environ:
    #if "RMSPROP" not in os.environ:
    #    out_folder = "output_cifar10"
    #else:
    #    out_folder = "output_cifar10_rmsprop"
    out_folder = "output_cifar10_deeper"
    for replicate in [0,1,2]:
        for dropout in [True, False]:
            for p in [0.1, 0.2, 0.3, 0.4, 0.5]:              
                for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
                    np.random.seed(replicate)
                    this_args = {"p":p, "nonlinearity": nonlinearity[1], "f1":16, "f2":32, "f3": 64, "f4":128}
                    if dropout:
                        this_args["dropout"] = True
                        out_file = "%s/p%f_%s_with_dense_dropout.%i" % (out_folder, p, nonlinearity[0], replicate)
                    else:
                        out_file = "%s/p%f_%s_with_dense.%i" % (out_folder, p, nonlinearity[0], replicate)
                    #if "RMSPROP" in os.environ:
                    #    this_args["rmsrop"] = True
                    if os.path.isfile("%s.txt" % out_file):
                        continue
                    train(
                        get_net(
                            get_deep_net_light_with_dense_for_cifar10(
                                this_args, custom_layer=MoreSkippableNonlinearityLayer),
                            (X_train, y_train, X_valid, y_valid), 
                            {"batch_size": 128}
                        ),
                        num_epochs=20,
                        data=(X_train, y_train, X_valid, y_valid),
                        out_file=out_file,
                        debug=False
                    )


# ----

# In[27]:

get_ipython().run_cell_magic(u'R', u'', u'files = c(\n    "p0.100000",\n    "p0.200000",\n    "p0.300000",\n    "p0.400000",\n    "p0.500000"\n)\nrainbows = rainbow(length(files))\nfor(i in 1:length(files)) {\n    df = read.csv(paste("output_more/",files[i],"_tanh_with_dense.txt",sep=""))\n    if(i == 1) {\n        plot(df$train_loss, type="l", col=rainbows[i], xlab="epoch", ylab="train loss")\n    } else {\n        lines(df$train_loss, col=rainbows[i])\n    }\n}\nfor(i in 1:length(files)) {\n    df = read.csv(paste("output_more/",files[i],"_tanh_with_dense.txt",sep=""))\n    if(i == 1) {\n        plot(df$valid_loss, type="l", col=rainbows[i], xlab="epoch", ylab="valid loss")\n    } else {\n        lines(df$valid_loss, col=rainbows[i])\n    }\n}')


# In[120]:

get_ipython().run_cell_magic(u'R', u'-w 800 -h 600', u'files = c(\n    "p0.100000",\n    "p0.200000",\n    "p0.300000",\n    "p0.400000",\n    "p0.500000"\n)\navg_df = function(filenames) {\n    df1 = read.csv(filenames[1])\n    df2 = read.csv(filenames[2])\n    df3 = read.csv(filenames[3])\n    return((df1+df2+df3) / 3)\n}\n\n\npar(mfrow=c(2,3))\ndf_baseline = avg_df(\n    c("output_more/p0.000000_tanh_with_dense.0.txt",\n    "output_more/p0.000000_tanh_with_dense.1.txt",\n    "output_more/p0.000000_tanh_with_dense.2.txt")\n)\nfor(i in 1:length(files)) {\n    df = avg_df(\n        c(paste("output_more/",files[i],"_tanh_with_dense.0.txt",sep=""),\n          paste("output_more/",files[i],"_tanh_with_dense.1.txt",sep=""),\n          paste("output_more/",files[i],"_tanh_with_dense.2.txt",sep=""))\n    )\n    df_drop = avg_df(\n        c(paste("output_more/",files[i],"_tanh_with_dense_dropout.0.txt",sep=""),\n          paste("output_more/",files[i],"_tanh_with_dense_dropout.1.txt",sep=""),\n          paste("output_more/",files[i],"_tanh_with_dense_dropout.2.txt",sep=""))\n    )\n    #df_baseline = read.csv(paste("output_more/",files[i],"_tanh_with_dense_dropout.txt",sep=""))\n    plot(df$valid_loss, type="l", col="blue", xlab="# epochs", ylab="valid loss", \n         ylim=c(0,0.25), main=paste(files[i],"(mnist)"))\n    lines(df_drop$valid_loss, col="red")\n    lines(df_baseline$valid_loss, col="black")\n    legend("topright", fill=c("blue", "red", "black"), legend=c("id", "dropout", "none"))\n}')


# In[122]:

get_ipython().run_cell_magic(u'R', u'-w 800 -h 600', u'files = c(\n    "p0.100000",\n    "p0.200000",\n    "p0.300000",\n    "p0.400000",\n    "p0.500000"\n)\navg_df = function(filenames) {\n    total_df = read.csv(filenames[1])\n    for(i in 2:length(filenames)) {\n        tmp = read.csv(filenames[i])\n        total_df = total_df + tmp\n    }\n    total_df = total_df / length(filenames)\n    return(total_df)\n}\n\n\npar(mfrow=c(2,3))\n\n# "output_more/p0.000000_relu_with_dense.1.txt"\n# has bad initialisation\n\ndf_baseline = avg_df(\n    c("output_more/p0.000000_relu_with_dense.0.txt",\n    "output_more/p0.000000_relu_with_dense.2.txt")\n)\nfor(i in 1:length(files)) {\n    df = avg_df(\n        c(paste("output_more/",files[i],"_relu_with_dense.0.txt",sep=""),\n          paste("output_more/",files[i],"_relu_with_dense.1.txt",sep=""),\n          paste("output_more/",files[i],"_relu_with_dense.2.txt",sep=""))\n    )\n    df_drop = avg_df(\n        c(paste("output_more/",files[i],"_relu_with_dense_dropout.0.txt",sep=""),\n          paste("output_more/",files[i],"_relu_with_dense_dropout.1.txt",sep=""),\n          paste("output_more/",files[i],"_relu_with_dense_dropout.2.txt",sep=""))\n    )\n    #df_baseline = read.csv(paste("output_more/",files[i],"_tanh_with_dense_dropout.txt",sep=""))\n    plot(df$valid_loss, type="l", col="blue", xlab="# epochs", ylab="valid loss", \n          ,main=paste(files[i],"(mnist)"))\n    lines(df_drop$valid_loss, col="red")\n    lines(df_baseline$valid_loss, col="black")\n    legend("topright", fill=c("blue", "red", "black"), legend=c("id", "dropout", "none"))\n}')


# ----

# **Examine the learning curves for the CIFAR10 experiments**

# In[123]:

get_ipython().run_cell_magic(u'R', u'-w 800 -h 600', u'files = c(\n    "p0.100000",\n    "p0.200000",\n    "p0.300000",\n    "p0.400000",\n    "p0.500000"\n)\navg_df = function(filenames) {\n    df1 = read.csv(filenames[1])\n    df2 = read.csv(filenames[2])\n    df3 = read.csv(filenames[3])\n    return((df1+df2+df3) / 3)\n}\n\n\npar(mfrow=c(2,3))\ndf_baseline = avg_df(\n    c("output_cifar10/p0.000000_tanh_with_dense.0.txt",\n    "output_cifar10/p0.000000_tanh_with_dense.1.txt",\n    "output_cifar10/p0.000000_tanh_with_dense.2.txt")\n)\nfor(i in 1:length(files)) {\n    df = avg_df(\n        c(paste("output_cifar10/",files[i],"_tanh_with_dense.0.txt",sep=""),\n          paste("output_cifar10/",files[i],"_tanh_with_dense.1.txt",sep=""),\n          paste("output_cifar10/",files[i],"_tanh_with_dense.2.txt",sep=""))\n    )\n    df_drop = avg_df(\n        c(paste("output_cifar10/",files[i],"_tanh_with_dense_dropout.0.txt",sep=""),\n          paste("output_cifar10/",files[i],"_tanh_with_dense_dropout.1.txt",sep=""),\n          paste("output_cifar10/",files[i],"_tanh_with_dense_dropout.2.txt",sep=""))\n    )\n    #df_baseline = read.csv(paste("output_more/",files[i],"_tanh_with_dense_dropout.txt",sep=""))\n    plot(df$valid_loss, type="l", col="blue", xlab="# epochs", ylab="valid loss", \n         ylim=c(1,2), main=paste(files[i],"(cifar10)"))\n    lines(df_drop$valid_loss, col="red")\n    lines(df_baseline$valid_loss, col="black")\n    legend("topright", fill=c("blue", "red", "black"), legend=c("id", "dropout", "none"))\n}')


# In[124]:

get_ipython().run_cell_magic(u'R', u'-w 800 -h 600', u'files = c(\n    "p0.100000",\n    "p0.200000",\n    "p0.300000",\n    "p0.400000",\n    "p0.500000"\n)\navg_df = function(filenames) {\n    n = 3\n    df1 = read.csv(filenames[1])\n    if( sum(is.nan(as.matrix(df1))) > 0 ) {\n        df1 = 0\n        n = n-1\n        write(paste("WARNING: NaNs in",filenames[1]), stderr())\n    }\n    df2 = read.csv(filenames[2])\n    if( sum(is.nan(as.matrix(df2))) > 0 ) {\n        df2 = 0\n        n = n-1\n        write(paste("WARNING: NaNs in",filenames[1]), stderr())\n    }\n    \n    df3 = read.csv(filenames[3])\n    if( sum(is.nan(as.matrix(df3))) > 0 ) {\n        df3 = 0\n        n = n-1\n        write(paste("WARNING: NaNs in",filenames[1]), stderr())\n    }\n    \n    return((df1+df2+df3) / n)\n}\n\n\npar(mfrow=c(2,3))\ndf_baseline = avg_df(\n    c("output_cifar10_tmp/p0.000000_relu_with_dense.0.txt",\n    "output_cifar10_tmp/p0.000000_relu_with_dense.1.txt",\n    "output_cifar10_tmp/p0.000000_relu_with_dense.2.txt")\n)\nfor(i in 1:length(files)) {\n    df = avg_df(\n        c(paste("output_cifar10_tmp/",files[i],"_relu_with_dense.0.txt",sep=""),\n          paste("output_cifar10_tmp/",files[i],"_relu_with_dense.1.txt",sep=""),\n          paste("output_cifar10_tmp/",files[i],"_relu_with_dense.2.txt",sep=""))\n    )\n    df_drop = avg_df(\n        c(paste("output_cifar10_tmp/",files[i],"_relu_with_dense_dropout.0.txt",sep=""),\n          paste("output_cifar10_tmp/",files[i],"_relu_with_dense_dropout.1.txt",sep=""),\n          paste("output_cifar10_tmp/",files[i],"_relu_with_dense_dropout.2.txt",sep=""))\n    )\n    #df_baseline = read.csv(paste("output_more/",files[i],"_tanh_with_dense_dropout.txt",sep=""))\n    plot(df$valid_loss, type="l", col="blue", xlab="# epochs", ylab="valid loss", \n         , main=paste(files[i],"(cifar10)"))\n    lines(df_drop$valid_loss, col="red")\n    lines(df_baseline$valid_loss, col="black")\n    legend("topright", fill=c("blue", "red", "black"), legend=c("id", "dropout", "none"))\n}')


# Was getting instabilities (well, non-learning) in training with relu, so changed to gain=relu. Still getting instabilities... but should we just move onto a deeper model anyway?

# In[107]:

get_ipython().run_cell_magic(u'R', u'', u'df=read.csv("output_cifar10_tmp/p0.000000_relu_with_dense.2.txt")\nplot(df$valid_loss, type="l")\n#df2 = read.csv("output_cifar10/p0.100000_relu_with_dense_dropout.1.txt")\n#lines(df2$valid_loss, col="red")')


# Try:
# * Experiment with PReLUs
# * Experiment with this technique + PReLUs
# * He init?

# -----

# Let's examine these models

# In[15]:

models2 = dict()
for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
    models2[ nonlinearity[0] ] = dict()
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        with open("output_more/p%f_%s_with_dense.model" % (p, nonlinearity[0])) as f:
            model = pickle.load(f)
        tmp_net = get_net(
            get_deep_net_light_with_dense({"p":0, "nonlinearity": nonlinearity[1]}, custom_layer=MoreSkippableNonlinearityLayer),
            (X_train, y_train, X_valid, y_valid), 
            {"batch_size": 128}
        )
        models2[ nonlinearity[0] ][p] = tmp_net
        set_all_param_values(tmp_net["l_out"], model)


# In[16]:

for nonlinearity in ["tanh", "relu"]:
    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print p, nonlinearity
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8,6))
        fig.tight_layout()
        for i in range(0,4*3):
            plt.subplot(4,3,i)
            plt.figure
            #plt.xlim(-20, 20)
            if nonlinearity == "tanh":
                plt.ylim(-1, 1)
            plt.ylabel("g(x)")
            plt.xlabel("x")
            plt.plot(
                models2[nonlinearity][0.0]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                models2[nonlinearity][0.0]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                "bo",
                models2[nonlinearity][p]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                models2[nonlinearity][p]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                "ro",
            )


# -----

# Ok, let's look at all the tanh models

# In[39]:

models = dict()
for nonlinearity in [("tanh", tanh), ("relu", rectify)]:
    models[ nonlinearity[0] ] = dict()
    for p in [0.75, 0.5, 0.25, 0.1, 0.0]:
        with open("output/p%f_%s.model" % (p, nonlinearity[0])) as f:
            model = pickle.load(f)
        tmp_net = get_net(
            get_deep_net_light({"p":0, "nonlinearity": nonlinearity[1]}),
            (X_train, y_train, X_valid, y_valid), 
            {"batch_size": 128, "max_norm": 10}
        )
        models[ nonlinearity[0] ][p] = tmp_net
        set_all_param_values(tmp_net["l_out"], model)


# Compare the tanh and relu models with the baseline

# In[12]:

get_ipython().run_cell_magic(u'R', u'', u'dfb = read.csv("output/p0.000000_tanh.txt")\ndf25 = read.csv("output/p0.250000_tanh.txt")\ndf50 = read.csv("output/p0.500000_tanh.txt")\ndf75 = read.csv("output/p0.750000_tanh.txt")\n\nmax_train_loss = max( c(max(dfb$train_loss), max(df25$train_loss), max(df50$train_loss), max(df75$train_loss) ) )\nmax_valid_loss = max( c(max(dfb$valid_loss), max(df25$valid_loss), max(df50$valid_loss), max(df75$valid_loss) ) )\n\npar(mfrow=c(1,2))\n# valid plots\nplot(dfb$valid_loss, type="l", xlab="# epochs", ylab="valid loss", col="blue", ylim=c(0, max_valid_loss))\nlines(df25$valid_loss, col="red")\nlines(df50$valid_loss, col="orange")\nlines(df75$valid_loss, col="green")\n# train plots\nplot(dfb$train_loss, type="l", xlab="# epochs", ylab="train loss", col="blue", ylim=c(0, max_train_loss))\nlines(df25$train_loss, col="red")\nlines(df50$train_loss, col="orange")\nlines(df75$train_loss, col="green")\n')


# In[13]:

get_ipython().run_cell_magic(u'R', u'', u'dfb = read.csv("output/p0.000000_relu.txt")\ndf25 = read.csv("output/p0.250000_relu.txt")\ndf50 = read.csv("output/p0.500000_relu.txt")\ndf75 = read.csv("output/p0.750000_relu.txt")\npar(mfrow=c(1,2))\n\nmax_train_loss = max( c(max(dfb$train_loss), max(df25$train_loss), max(df50$train_loss), max(df75$train_loss) ) )\nmax_valid_loss = max( c(max(dfb$valid_loss), max(df25$valid_loss), max(df50$valid_loss), max(df75$valid_loss) ) )\n\n# valid plots\nplot(dfb$valid_loss, type="l", xlab="# epochs", ylab="valid loss", col="blue", ylim=c(0, max_valid_loss))\nlines(df25$valid_loss, col="red")\nlines(df50$valid_loss, col="orange")\nlines(df75$valid_loss, col="green")\n# train plots\nplot(dfb$train_loss, type="l", xlab="# epochs", ylab="train loss", col="blue", ylim=c(0, max_train_loss))\nlines(df25$train_loss, col="red")\nlines(df50$train_loss, col="orange")\nlines(df75$train_loss, col="green")\n')


# Compare dropout without dropout

# In[37]:

get_ipython().run_cell_magic(u'R', u'', u'files = c(\n    "p0.100000_tanh",\n    "p0.250000_tanh",\n    "p0.500000_tanh",\n    "p0.750000_tanh",\n    "p0.100000_relu",\n    "p0.250000_relu",\n    "p0.500000_relu",\n    "p0.750000_relu"\n)\n\nplot_result = function(filename) {\n    par(mfrow=c(1,2))\n    df0 = read.csv(paste("output/",filename,".txt",sep=""))\n    df0d = read.csv(paste("output/",filename,"_dropout.txt",sep=""))\n    max_train_loss = max( c(max(df0$train_loss), max(df0d$train_loss)) )\n    max_valid_loss = max( c(max(df0$valid_loss), max(df0d$valid_loss)) )\n    plot(df0$train_loss, type="l", col="blue", xlab="# epochs",\n         ylab="train loss", main=filename, ylim=c(0, max_train_loss))\n    lines(df0d$train_loss, type="l", col="red")\n    plot(df0$valid_loss, type="l", col="blue", xlab="# epochs",\n         ylab="valid loss", main=filename, ylim=c(0, max_valid_loss))\n    lines(df0d$valid_loss, type="l", col="red")    \n}\n\n\nfor( i in 1:length(files)) {\n    plot_result(files[i])\n}')


# In[40]:

for nonlinearity in ["tanh", "relu"]:
    for p in [0.1, 0.25, 0.5, 0.75]:
        print p, nonlinearity
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8,6))
        fig.tight_layout()
        for i in range(0,4*3):
            plt.subplot(4,3,i)
            plt.figure
            #plt.xlim(-20, 20)
            if nonlinearity == "tanh":
                plt.ylim(-1, 1)
            plt.ylabel("g(x)")
            plt.xlabel("x")
            plt.plot(
                models[nonlinearity][0.0]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                models[nonlinearity][0.0]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                "bo",
                models[nonlinearity][p]["outs_without_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                models[nonlinearity][p]["outs_with_nonlinearity"](X_train.get_value()[0:100])[i].flatten(),
                "ro",
            )


# Observations:
# * For relu, trick seems to push the points away from the negative part (x < 0). So this is in a way doing the reverse of encouraging sparsity.
# * For tanh, trick is pushing units away from saturation regime.
# * $\frac{\partial L}{\partial g(Wx)} \cdot \frac{\partial g(Wx)}{\partial Wx} \cdot \frac{\partial Wx}{\partial W} = \frac{\partial L}{\partial W}$
#    * Let's examine the case when we're using a sigmoid. When it is identity, $\frac{\partial g(Wx)}{\partial Wx} \cdot \frac{\partial Wx}{\partial W} = 1 \cdot x$, so if $x$ is big then the gradient will be big. When the sigmoid is used, it becomes $\frac{\partial sigm(Wx)}{\partial x} \cdot x$. If $x$ is big, then $\frac{\partial sigm(Wx)}{\partial x}$ will be very small (saturation regime). So when $x$ is big we alternate between big and tiny gradients, the big gradients encouraging $x$ to not be near the saturating points. The big gradients force us to use norm clipping.
#    * For relu, when $x > 0$, $\frac{\partial g(Wx)}{\partial Wx} \cdot \frac{\partial Wx}{\partial W} = 1 \cdot x$ no matter if relu or identity is used. If $x < 0$, then for relu $\frac{\partial g(Wx)}{\partial Wx} \cdot \frac{\partial Wx}{\partial W} = 0 \cdot x$, but then if identity is used $\frac{\partial g(Wx)}{\partial Wx} \cdot \frac{\partial Wx}{\partial W} = 1 \cdot x$, giving it a gradient and potentially "un-killing" neurons.

# ----

# In[ ]:



