import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from lasagne.regularization import *

def _remove_trainable(layer):
    for key in layer.params:
        layer.params[key].remove('trainable')

def _residual_block(layer, n_out_channels, prefix, stride=1, nonlinearity=rectify):
    """
    residual block
    :param layer:
    :param n_out_channels:
    :param prefix:
    :param stride:
    :param nonlinearity:
    :return:
    """
    conv = layer
    if stride > 1:
        layer = Pool2DLayer(layer, pool_size=1, stride=stride, mode="average_inc_pad")
    if (n_out_channels != layer.output_shape[1]):
        diff = n_out_channels - layer.output_shape[1]
        if diff % 2 == 0:
            width_tp = ((diff / 2, diff / 2),)
        else:
            width_tp = (((diff / 2) + 1, diff / 2),)
        layer = pad(layer, batch_ndim=1, width=width_tp)
    conv = Conv2DLayer(conv,
                       num_filters=n_out_channels,
                       filter_size=(3, 3),
                       stride=(stride, stride),
                       pad=(1, 1),
                       nonlinearity=linear,
                       W=HeNormal(gain="relu"),
                       b=Constant(0.))
    conv = BatchNormLayer(conv,
                          beta=Constant(0.),
                          gamma=Constant(1.))
    conv = NonlinearityLayer(conv, nonlinearity=nonlinearity)
    conv = Conv2DLayer(conv,
                       num_filters=n_out_channels,
                       filter_size=(3, 3),
                       stride=(1, 1),
                       pad=(1, 1),
                       nonlinearity=linear,
                       W=HeNormal(gain="relu"),
                       b=Constant(0.))
    conv = BatchNormLayer(conv,
                          beta=Constant(0.),
                          gamma=Constant(1.))
    return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity=nonlinearity)


def _resnet_2x4(l_in, nf=[32, 64, 128, 256], N=2):
    """
    :param quadrant:
    :param dd: for param sharing purposes (deprecated)
    :param first_time:
    :param nf:
    :param N:
    :param dropout_p:
    :return:
    """
    assert len(nf) == 4 # this is a 4-block resnet
    layer = Conv2DLayer(l_in,
                        num_filters=nf[0],
                        filter_size=7,
                        stride=2,
                        nonlinearity=rectify,
                        pad='same',
                        W=HeNormal(gain="relu"),
                        b=Constant(0.))
    layer = MaxPool2DLayer(layer, pool_size=3, stride=2)
    for i in range(N):
        layer = _residual_block(layer, nf[0], prefix="a%i" % i)
    layer = _residual_block(layer, nf[1], prefix="aa", stride=2)
    for i in range(N):
        layer = _residual_block(layer, nf[1], prefix="b%i" % i)
    layer = _residual_block(layer, nf[2], prefix="bb%", stride=2)
    for i in range(N):
        layer = _residual_block(layer, nf[2], prefix="c%i" % i)
    layer = _residual_block(layer, nf[3], prefix="cc", stride=2)
    for i in range(N):
        layer = _residual_block(layer, nf[3], prefix="dd%i" % i)
    layer = Pool2DLayer(layer, pool_size=layer.output_shape[-1], stride=1, mode="average_inc_pad")
    layer = FlattenLayer(layer)
    return layer

def _add_pois(layer, num_classes, end_nonlinearity, tau):
    from scipy.misc import factorial
    layer = DenseLayer(layer, num_units=1, nonlinearity=end_nonlinearity)
    l_copy = DenseLayer(layer, num_units=num_classes, nonlinearity=linear)
    l_copy.W.set_value( np.ones((1,num_classes)).astype("float32") )
    _remove_trainable(l_copy)
    l_pois = ExpressionLayer(l_copy, lambda x: ((c*T.log(x)) - x - T.log(cf)) / tau )
    c = np.asarray([[(i+1) for i in range(0, num_classes)]], dtype="float32")
    cf = factorial(c)
    l_softmax = NonlinearityLayer(l_pois, nonlinearity=softmax)    
    return l_softmax

def resnet_2x4_adience(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=8, nonlinearity=softmax)
    return layer

def resnet_2x4_adience_pois(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = _add_pois(layer, end_nonlinearity=args["end_nonlinearity"], num_classes=8, tau=args["tau"])
    return layer

def resnet_2x4_dr(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=5, nonlinearity=softmax)
    return layer

def resnet_2x4_dr_pois(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = _add_pois(layer, end_nonlinearity=args["end_nonlinearity"], num_classes=5, tau=args["tau"])
    return layer
    
if __name__ == '__main__':

    l_in = InputLayer((None, 3, 224, 224))
    _, l_out = _resnet_2x4(l_in, {}, True)
    for layer in get_all_layers(l_out):
        print layer, "", layer.output_shape
