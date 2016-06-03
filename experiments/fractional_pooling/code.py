
# coding: utf-8

# In[38]:

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


# In[39]:

train_data, valid_data, _ = hp.load_mnist("../../data/mnist.pkl.gz")
X_train, y_train = train_data
X_valid, y_valid = valid_data


# In[73]:

X_train, y_train = X_train.astype("float32"), y_train.astype("int32")
X_valid, y_valid = X_valid.astype("float32"), y_valid.astype("int32")


# In[22]:

_srng = T.shared_randomstreams.RandomStreams()

def theano_shuffled(input):
    n = input.shape[0]

    shuffled = T.permute_row_elements(input.T, _srng.permutation(n=n)).T
    return shuffled

class FractionalPool2DLayer(Layer):
    """
    Fractional pooling as described in http://arxiv.org/abs/1412.6071
    Only the random overlapping mode is currently implemented.
    """
    def __init__(self, incoming, ds, pool_function=T.max, **kwargs):
        super(FractionalPool2DLayer, self).__init__(incoming, **kwargs)
        if type(ds) is not tuple:
            raise ValueError("ds must be a tuple")
        if (not 1 <= ds[0] <= 2) or (not 1 <= ds[1] <= 2):
            raise ValueError("ds must be between 1 and 2")
        self.ds = ds  # a tuple
        if len(self.input_shape) != 4:
            raise ValueError("Only bc01 currently supported")
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape) # copy / convert to mutable list
        output_shape[2] = int(np.ceil(float(output_shape[2]) / self.ds[0]))
        output_shape[3] = int(np.ceil(float(output_shape[3]) / self.ds[1]))

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        _, _, n_in0, n_in1 = self.input_shape
        _, _, n_out0, n_out1 = self.output_shape

        # Variable stride across the input creates fractional reduction
        a = theano.shared(
            np.array([2] * (n_in0 - n_out0) + [1] * (2 * n_out0 - n_in0)))
        b = theano.shared(
            np.array([2] * (n_in1 - n_out1) + [1] * (2 * n_out1 - n_in1)))

        # Randomize the input strides
        a = theano_shuffled(a)
        b = theano_shuffled(b)

        # Convert to input positions, starting at 0
        a = T.concatenate(([0], a[:-1]))
        b = T.concatenate(([0], b[:-1]))
        a = T.cumsum(a)
        b = T.cumsum(b)

        # Positions of the other corners
        c = T.clip(a + 1, 0, n_in0 - 1)
        d = T.clip(b + 1, 0, n_in1 - 1)

        # Index the four positions in the pooling window and stack them
        temp = T.stack(input[:, :, a, :][:, :, :, b],
                       input[:, :, c, :][:, :, :, b],
                       input[:, :, a, :][:, :, :, d],
                       input[:, :, c, :][:, :, :, d])

        return self.pool_function(temp, axis=0)


# In[110]:

def fractional_net():
    l_in = InputLayer( (None, 1, 28, 28) )
    l_conv1 = Conv2DLayer(l_in, num_filters=16, filter_size=3)
    l_mp1 = FractionalPool2DLayer(l_conv1, ds=(1.5,1.5))
    l_conv2 = Conv2DLayer(l_mp1, num_filters=32, filter_size=3)
    l_mp2 = FractionalPool2DLayer(l_conv2, ds=(1.5,1.5))
    l_conv3 = Conv2DLayer(l_mp2, num_filters=40, filter_size=3)
    l_mp3 = FractionalPool2DLayer(l_conv3, ds=(1.5,1.5))
    l_conv4 = Conv2DLayer(l_mp3, num_filters=48, filter_size=3)
    l_mp3 = FractionalPool2DLayer(l_conv4, ds=(1.5,1.5))
    l_dense = DenseLayer(l_conv4, num_units=10, nonlinearity=softmax)
    return l_dense


# In[88]:

def normal_net():
    l_in = InputLayer( (None, 1, 28, 28) )
    l_conv1 = Conv2DLayer(l_in, num_filters=32, filter_size=4)
    l_mp1 = MaxPool2DLayer(l_conv1, pool_size=(2,2))
    l_conv2 = Conv2DLayer(l_mp1, num_filters=48, filter_size=3)
    l_mp2 = MaxPool2DLayer(l_conv2, pool_size=(2,2))
    l_conv3 = Conv2DLayer(l_mp2, num_filters=64, filter_size=3)
    l_mp3 = MaxPool2DLayer(l_conv3, pool_size=(2,2))
    l_dense = DenseLayer(l_mp3, num_units=10, nonlinearity=softmax)
    return l_dense


# In[59]:

X = l_in.input_var


# In[111]:

l_out = fractional_net()
for layer in get_all_layers(l_out):
    print layer, layer.output_shape
print "number of params:", count_params(l_out)


# In[89]:

l_out = normal_net()
for layer in get_all_layers(l_out):
    print layer, layer.output_shape
print "number of params:", count_params(l_out)


# In[77]:

X = T.tensor4('X')
y = T.ivector('y')
l_out = fractional_net()
# ----
net_out = get_output(l_out, X)
loss = categorical_crossentropy(net_out, y).mean()
params = get_all_params(l_out, trainable=True)
grads = T.grad(loss, params)
learning_rate = 0.01
momentum = 0.9
updates = nesterov_momentum(grads, params, learning_rate=learning_rate, momentum=momentum)


# In[78]:

train_fn = theano.function([X,y], loss, updates=updates)


# In[79]:

t0 = time()
bs = 32
n_batches = X_train.shape[0] // bs

print "epochs", "time"
num_epochs=10
for epoch in range(0, num_epochs):
    losses=[]
    for b in range(0, n_batches):
        losses.append( train_fn(X_train[b*bs:(b+1)*bs], y_train[b*bs:(b+1)*bs]) )
    print np.mean(losses), time()-t0


# In[ ]:



