import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from lasagne.regularization import *

def residual_block(layer, n_out_channels, prefix, stride=1, dd={}, nonlinearity=rectify):
    """
    residual block
    :param layer:
    :param n_out_channels:
    :param prefix:
    :param stride:
    :param dd: for param sharing (deprecated)
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
                       W=HeNormal(gain="relu") if prefix + "_1" not in dd else dd[prefix + "_1"].W,
                       b=Constant(0.) if prefix + "_1" not in dd else dd[prefix + "_1"].b)
    if prefix + "_1" not in dd:
        dd[prefix + "_1"] = conv
        #print prefix + "_1"
    conv = BatchNormLayer(conv,
                          beta=Constant(0.) if prefix + "_bn1" not in dd else dd[prefix + "_bn1"].beta,
                          gamma=Constant(1.) if prefix + "_bn1" not in dd else dd[prefix + "_bn1"].gamma)
    if prefix + "_bn1" not in dd:
        dd[prefix + "_bn1"] = conv
        #print prefix + "_bn1"
    conv = NonlinearityLayer(conv, nonlinearity=nonlinearity)
    conv = Conv2DLayer(conv,
                       num_filters=n_out_channels,
                       filter_size=(3, 3),
                       stride=(1, 1),
                       pad=(1, 1),
                       nonlinearity=linear,
                       W=HeNormal(gain="relu") if prefix + "_2" not in dd else dd[prefix + "_2"].W,
                       b=Constant(0.) if prefix + "_2" not in dd else dd[prefix + "_2"].b)
    if prefix + "_2" not in dd:
        dd[prefix + "_2"] = conv
        #print prefix + "_2"
    conv = BatchNormLayer(conv,
                          beta=Constant(0.) if prefix + "_bn2" not in dd else dd[prefix + "_bn2"].beta,
                          gamma=Constant(1.) if prefix + "_bn2" not in dd else dd[prefix + "_bn2"].gamma)
    if prefix + "_bn2" not in dd:
        dd[prefix + "_bn2"] = conv
        #print prefix + "_bn2"
    return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity=nonlinearity)


def resnet_2x4(quadrant, dd, first_time, nf=[32, 64, 128, 256], N=2):
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
    for i in range(N):
        layer = residual_block(layer, nf[0], prefix="a%i" % i, dd=dd)
    layer = residual_block(layer, nf[1], prefix="aa", stride=2, dd=dd)
    for i in range(N):
        layer = residual_block(layer, nf[1], prefix="b%i" % i, dd=dd)
    layer = residual_block(layer, nf[2], prefix="bb%", stride=2, dd=dd)
    for i in range(N):
        layer = residual_block(layer, nf[2], prefix="c%i" % i, dd=dd)
    layer = residual_block(layer, nf[3], prefix="cc", stride=2, dd=dd)
    for i in range(N):
        layer = residual_block(layer, nf[3], prefix="dd%i" % i, dd=dd)
    layer = Pool2DLayer(layer, pool_size=layer.output_shape[-1], stride=1, mode="average_inc_pad")
    layer = FlattenLayer(layer)
    if first_time:
        return dd, layer
    else:
        return layer

if __name__ == '__main__':

    l_in = InputLayer((None, 3, 224, 224))
    _, l_out = resnet_2x4(l_in, {}, True)
    for layer in get_all_layers(l_out):
        print layer, "", layer.output_shape