import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from lasagne.regularization import *

def light_net(args):
    layer = InputLayer((None, 1, 28, 28))
    for i in range(3):
        layer = Conv2DLayer(layer, num_filters=(i+1)*16, filter_size=3)
        layer = MaxPool2DLayer(layer, pool_size=2)
    layer = DenseLayer(layer, num_units=10, nonlinearity=softmax)
    return layer

if __name__ == '__main__':
    print light_net({})