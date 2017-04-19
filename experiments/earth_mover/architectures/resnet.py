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
import os
sys.path.append(os.environ["EARTH_MOVER"])
from layers import TauLayer

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

class DivLayer(MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(DivLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return self.input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        numerator, denominator = inputs
        denominator = T.addbroadcast(denominator, 1) # in case denom is of shape (bs,1)
        return numerator / denominator

def _add_pois(layer, num_classes, end_nonlinearity, tau, tau_mode="non_learnable"):
    """
    :tau_mode:
     learnable = we learn tau directly (but it is the same for all inputs)
     sigm_learnable = we learn tau inside a sigmoid (but it is the same for all inputs)
     unlearnable = tau is fixed (obviously, the same for all inputs)
     fn_learnable = we learn a function sigm(T(x)) (changes depending on x)
    """
    assert tau_mode in ["learnable", "non_learnable", "sigm_learnable", "fn_learnable", "fn_learnable_fixed"]
    if tau_mode == "fn_learnable":
        raise NotImplementedError("fn_learnable (of as before 18/04/2017) incorrectly learned a tau for " + \
                                  "every one of the k classes, as opposed to just one tau. Please use fn_learnable_fixed")
    from scipy.misc import factorial
    #layer.name = "avg_pool"
    l_fx = DenseLayer(layer, num_units=1, nonlinearity=end_nonlinearity)
    l_fx.tag = "fx"
    l_copy = DenseLayer(l_fx, num_units=num_classes, nonlinearity=linear)
    l_copy.W.set_value( np.ones((1,num_classes)).astype("float32") )
    _remove_trainable(l_copy)
    c = np.asarray([[(i+1) for i in range(0, num_classes)]], dtype="float32")
    cf = factorial(c)
    if tau_mode == "non_learnable":
        l_pois = ExpressionLayer(l_copy, lambda x: ((c*T.log(x)) - x - T.log(cf)) / tau )
    elif tau_mode in ["learnable", "sigm_learnable"]:
        l_pois = ExpressionLayer(l_copy, lambda x: ((c*T.log(x)) - x - T.log(cf)) )
        if tau_mode == "learnable":
            fn = linear
        elif tau_mode == "sigm_learnable":
            fn = sigmoid
        l_pois = TauLayer(l_pois, tau=lasagne.init.Constant(tau), bias=0., nonlinearity=fn)
    elif tau_mode == "fn_learnable_fixed":
        l_exp = ExpressionLayer(l_copy, lambda x: ((c*T.log(x)) - x - T.log(cf)))
        # this is the T(x) layer that we learn
        l_tau_pre = DenseLayer(layer, num_units=1, nonlinearity=softplus) # prior to 18/04: num_units=num_classes
        l_tau = ExpressionLayer(l_tau_pre, lambda x: 1.0 / (1.0 + x))
        l_tau.name = "tau_fn"
        # then we compute h(x) / T(x)
        l_div = DivLayer((l_exp,l_tau))
        l_pois = l_div
    l_softmax = NonlinearityLayer(l_pois, nonlinearity=softmax)
    return l_softmax

def _add_binom(layer, num_classes, tau, tau_mode):
    assert tau_mode in ["non_learnable", "sigm_learnable", "fn_learnable", "fn_learnable_fixed"]
    if tau_mode == "fn_learnable":
        raise NotImplementedError("fn_learnable (of as before 18/04/2017) incorrectly learned a tau for " + \
                                  "every one of the k classes, as opposed to just one tau. Please use fn_learnable_fixed")
    # NOTE: weird bug. so this is numerically unstable when
    # deterministic=True, but i am unable to reproduce it
    # outside of using this with my resnet
    # so: i added eps, and added clip
    from scipy.special import binom
    k = num_classes
    l_sigm = DenseLayer(layer, num_units=1, nonlinearity=sigmoid, W=HeNormal(gain="relu"), b=Constant(0.))
    l_copy = DenseLayer(l_sigm, num_units=k, nonlinearity=linear)
    l_copy.W.set_value( np.ones((1,k)).astype("float32") )
    _remove_trainable(l_copy)
    c = np.asarray([[(i) for i in range(0, k)]], dtype="float32")
    binom_coef = binom(k-1, c).astype("float32")
    ### NOTE: NUMERICALLY UNSTABLE ###
    eps = 1e-6
    l_logf = ExpressionLayer( l_copy, lambda px: (T.log(binom_coef) + (c*T.log(px+eps)) + ((k-1-c)*T.log(1.-px+eps))) )
    if tau_mode == "non_learnable":
        if tau != 1:
            l_logf = ExpressionLayer(l_logf, lambda px: px / tau)
    else:
        if tau_mode == "sigm_learnable":
            l_logf = TauLayer(l_logf, tau=lasagne.init.Constant(tau), bias=0., nonlinearity=sigmoid)
        elif tau_mode == "fn_learnable_fixed":
            l_h = DenseLayer(layer, num_units=1, nonlinearity=softplus)
            l_h = ExpressionLayer(l_h, lambda x: 1.0 / (1.0 + x)) # fc(k)
            # then we compute h(x) / T(x)
            l_logf = DivLayer((l_logf,l_tau))        
    l_logf = NonlinearityLayer(l_logf, nonlinearity=softmax)
    return l_logf

def _add_binom_scap(layer, num_classes):
    from scipy.special import binom
    k = num_classes
    l_pre = DenseLayer(layer, num_units=k, nonlinearity=linear)
    l_sigm = DenseLayer(l_pre, num_units=1, nonlinearity=sigmoid)
    l_copy = DenseLayer(l_sigm, num_units=k, nonlinearity=linear)
    l_copy.W.set_value( np.ones((1,k)).astype("float32") )
    _remove_trainable(l_copy)
    c = np.asarray([[(i) for i in range(0, k)]], dtype="float32")
    binom_coef = binom(k-1, c).astype("float32")
    ### NOTE: NUMERICALLY UNSTABLE ###
    eps = 1e-6
    #l_logf = ExpressionLayer( l_copy, lambda px: (c*T.log(px)) + ((k-1-c)*T.log(1.-px)) )
    #l_logf = ExpressionLayer(l_logf, lambda px: T.exp(px))
    #l_logf = ExpressionLayer(l_logf, lambda px: binom_coef*px)
    l_logf = ExpressionLayer( l_copy, lambda px: T.log(binom_coef) + (c*T.log(px+eps)) + ((k-1-c)*T.log(1.-px+eps)) )
    l_logf = NonlinearityLayer(l_logf, nonlinearity=softmax)
    return l_logf


#### TEST ####
def _add_binom_test(layer, num_classes):
    from scipy.special import binom
    k = num_classes
    l_copy = DenseLayer(layer, num_units=k, nonlinearity=linear)
    l_copy.W.set_value( np.ones((1,k)).astype("float32") )
    _remove_trainable(l_copy)
    c = np.asarray([[(i) for i in range(0, k)]], dtype="float32")
    binom_coef = binom(k-1, c).astype("float32")
    print binom_coef
    l_logf = ExpressionLayer( l_copy, lambda px: binom_coef*T.exp( (c*T.log(px)) + ((k-1-c)*T.log(1.-px)) ) )
    #l_explogf = ExpressionLayer(l_logf, lambda x: T.exp(x))
    return l_logf
#### TEST ####

def resnet_2x4_adience(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=8, nonlinearity=softmax)
    return layer

def resnet_2x4_adience_test1(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=1, nonlinearity=softmax)
    return layer

def resnet_2x4_adience_tau(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=8, nonlinearity=linear)
    layer = TauLayer(layer, tau=lasagne.init.Constant(args["tau"]), bias=0.0)
    layer = NonlinearityLayer(layer, nonlinearity=softmax)
    return layer

def resnet_2x4_adience_pois(args):
    """
    NOTE: this layer does not have the same # of params as
    resnet_2x4_adience. This is because instead of k units after
    the pooling layer, we simply have 1 unit. This problem is
    rectified in the method `resnet_2x4_adience_pois_scap`
    (scap = same capacity)
    """
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = _add_pois(layer, end_nonlinearity=args["end_nonlinearity"], num_classes=8, tau=args["tau"], tau_mode=args["tau_mode"])
    return layer

def resnet_2x4_adience_pois_scap(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=8, nonlinearity=linear)
    layer = _add_pois(layer, end_nonlinearity=args["end_nonlinearity"], num_classes=8, tau=args["tau"], tau_mode=args["tau_mode"])
    return layer

def resnet_2x4_adience_pois_scap_relu(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=8, nonlinearity=rectify)
    layer = _add_pois(layer, end_nonlinearity=args["end_nonlinearity"], num_classes=8, tau=args["tau"], tau_mode=args["tau_mode"])
    return layer


def resnet_2x4_adience_binom(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = _add_binom(layer, num_classes=8, tau=args["tau"], tau_mode=args["tau_mode"])
    return layer


def resnet_2x4_dr(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=5, nonlinearity=softmax)
    return layer

def resnet_2x4_dr_tau(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = DenseLayer(layer, num_units=5, nonlinearity=linear)
    layer = TauLayer(layer, tau=lasagne.init.Constant(args["tau"]), bias=0.0)
    layer = NonlinearityLayer(layer, nonlinearity=softmax)
    return layer

def resnet_2x4_dr_pois(args):
    layer = InputLayer((None,3,224,224))
    layer = _resnet_2x4(layer)
    layer = _add_pois(layer, end_nonlinearity=args["end_nonlinearity"], num_classes=5, tau=args["tau"])
    return layer
    
if __name__ == '__main__':

    #l_in = InputLayer((None, 3, 224, 224))
    #_, l_out = _resnet_2x4(l_in, {}, True)
    #for layer in get_all_layers(l_out):
    #    print layer, "", layer.output_shape

    """
    from lasagne.utils import floatX
    import sys
    sys.path.append("..")
    from layers import TauLayer
    l_in = InputLayer((None,2))
    l_tau = TauLayer(l_in, tau=lasagne.init.Constant(1.))
    print get_all_params(l_tau)
    X = T.fmatrix('X')

    inp = np.asarray([[10.0,5.0],[20.0,10.0]])
    print get_output(l_tau,X).eval({X:inp.astype("float32")})
    """

    """
    l_out_1 = resnet_2x4_adience({})
    l_out_2 = resnet_2x4_adience_test1({})
    l_out_3 = resnet_2x4_adience_pois({"tau":1.0, "tau_mode":"non_learnable", "end_nonlinearity":lasagne.nonlinearities.softplus})

    print "resnet adience (8 classes):", count_params(l_out_1, trainable=True)
    print "resnet adience (1 class test):", count_params(l_out_2, trainable=True)
    print "resnet adience (8 classes) (pois extension not learning tau):", count_params(l_out_3,trainable=True)

    assert count_params(l_out_2, trainable=True) == count_params(l_out_3,trainable=True)
    """

    """
    l_out_1 = resnet_2x4_dr({})
    l_out_2 = resnet_2x4_adience_test1({}) # same
    l_out_3 = resnet_2x4_dr_pois({"tau":1.0, "tau_mode":"non_learnable", "end_nonlinearity":lasagne.nonlinearities.softplus})

    print "resnet dr (1 classes):", count_params(l_out_1, trainable=True)
    print "resnet dr (1 class test):", count_params(l_out_2, trainable=True)
    print "resnet dr (1 classes) (pois extension not learning tau):", count_params(l_out_3,trainable=True)

    assert count_params(l_out_2, trainable=True) == count_params(l_out_3,trainable=True)
    """

    """
    l_in = InputLayer((None,1))
    l_out = _add_binom_test(l_in, num_classes=8)
    X = T.fmatrix('X')
    net_out = get_output(l_out, X, deterministic=True)
    fake_x = np.random.random(size=(100,1))
    print fake_x
    pdists = net_out.eval({X:fake_x.astype("float32")})
    print np.sum(pdists,axis=1) # want all to be == 1.
    """

    l_out_learntau = resnet_2x4_adience_pois({"tau":1.0, "tau_mode":"learnable", "end_nonlinearity":lasagne.nonlinearities.softplus})
    #l_out_learntaufn = resnet_2x4_adience_pois({"tau":1.0, "tau_mode":"fn_learnable", "end_nonlinearity":lasagne.nonlinearities.softplus}) # disabled for now
    l_out_learntaufn_fixed = resnet_2x4_adience_pois({"tau":1.0, "tau_mode":"fn_learnable_fixed", "end_nonlinearity":lasagne.nonlinearities.softplus}) # disabled for now    
    l_out_learntau_scap = resnet_2x4_adience_pois_scap({"tau":1.0, "tau_mode":"learnable", "end_nonlinearity":lasagne.nonlinearities.softplus})
    l_out_vanilla = resnet_2x4_adience({})
    
    print "adience learn tau", count_params(l_out_learntau)
    print "adience learn tau fn fixed", count_params(l_out_learntaufn_fixed)
    print "adience learn tau 'scap'", count_params(l_out_learntau_scap)
    print "adience vanilla", count_params(l_out_vanilla)
