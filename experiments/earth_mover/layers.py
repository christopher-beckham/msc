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

if __name__ == '__main__':
    # test to see if this works
    l_in = InputLayer((None, 5))
    l_uro = UpperRightOnesLayer(l_in)
    X = T.fmatrix('X')
    input = np.asarray([[1,2,3,4,5]]).astype("float32")
    get_out = get_output(l_uro, X)
    assert np.all( get_out.eval({X: input})[0] == np.asarray([1,3,6,10,15]).astype("float32") )