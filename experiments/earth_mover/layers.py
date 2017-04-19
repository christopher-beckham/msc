import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from lasagne.regularization import *

class UpperRightOnesLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(UpperRightOnesLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        # construct the upper triangular matrix
        self.W = np.ones((num_inputs, num_inputs), dtype="float32")
        for k in range(0, num_inputs):
            self.W[k][0:k] = 0

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_inputs)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        result = T.dot(input, self.W)
        return result

class OrdinalSubtractLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(OrdinalSubtractLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        # construct the matrix
        self.W = np.zeros((num_inputs, num_inputs+1), dtype="float32")
        self.W[0,0]=1
        for k in range(1, num_inputs):
            self.W[k-1,k] = -1
            self.W[k,k] = 1
        self.W[num_inputs-1,num_inputs] = 1
        # construct the bias row vector
        self.b = np.zeros((1, num_inputs+1), dtype="float32")
        self.b[0, num_inputs] = 1
        #print self.W
        #print self.b

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_inputs+1)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        result = T.dot(input, self.W)
        result = T.abs_(self.b - result)
        #eps = 0.01
        #result = result + eps
        return result

class TauLayer(Layer):
    """
    Divide the input by b + g(tau), where b is a pre-specified bias
    and g is a pre-specified nonlinearity.
    """
    def __init__(self, incoming, tau, bias=1.0, nonlinearity=linear, **kwargs):
        super(TauLayer, self).__init__(incoming, **kwargs)
        self.tau = self.add_param(tau, (1,), name='tau', regularizable=False)
        self.bias = bias
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        result = input / (self.bias + self.nonlinearity(self.tau))
        return result
    
if __name__ == '__main__':
    # test to see if this works
    l_in = InputLayer((None, 5))
    l_uro = UpperRightOnesLayer(l_in)
    X = T.fmatrix('X')
    input = np.asarray([[1,2,3,4,5]]).astype("float32")
    get_out = get_output(l_uro, X)
    assert np.all( get_out.eval({X: input})[0] == np.asarray([1,3,6,10,15]).astype("float32") )
