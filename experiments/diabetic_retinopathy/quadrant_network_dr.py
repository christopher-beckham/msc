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
sys.path.append("../../modules/")
import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float
import cPickle as pickle
import os
from time import time
from keras.preprocessing.image import ImageDataGenerator
import pdb
from itertools import cycle

class SimpleOrdinalSubtractLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(SimpleOrdinalSubtractLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        # construct the matrix
        self.W = np.zeros((num_inputs, num_inputs), dtype="float32")
        self.W[0,0]=1
        for k in range(1, num_inputs):
            self.W[k-1,k] = -1
            self.W[k,k] = 1
        print self.W

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_inputs)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        result = T.dot(input, self.W)
        return result



class DivisionLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(DivisionLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_inputs)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
            
        result = input
        last_col = T.addbroadcast(result[:,-1::], 1)
            
        return result / last_col

    
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
        print self.W
        print self.b

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

def sigm_scaled(x):
    k = 5 # 5 classes
    return sigmoid(x)*(k-1)

def qwk(predictions, targets, num_classes=5):
    w = np.ones((num_classes,num_classes))
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            w[i,j] = (i-j)**2
    numerator = T.dot(targets.T, predictions)
    denominator = T.outer(targets.sum(axis=0), predictions.sum(axis=0))
    denominator = denominator / T.sum(numerator)
    qwk = (T.sum(numerator*w) / T.sum(denominator*w))
    return qwk

def qwk_normw(predictions, targets, num_classes=5):
    w = np.ones((num_classes,num_classes))
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            w[i,j] = ( (1.0*i-j)**2 ) / ((num_classes-1)**2)
    numerator = T.dot(targets.T, predictions)
    denominator = T.outer(targets.sum(axis=0), predictions.sum(axis=0))
    denominator = denominator / T.sum(numerator)
    qwk = (T.sum(numerator*w) / T.sum(denominator*w))
    return qwk

def kappa(predictions, targets, num_classes=5):
    w = np.ones((num_classes,num_classes))
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            w[i,j] = 1
            if i==j:
                w[i,j]=0
    numerator = T.dot(targets.T, predictions)
    denominator = T.outer(targets.sum(axis=0), predictions.sum(axis=0))
    denominator = denominator / T.sum(numerator)
    qwk = (T.sum(numerator*w) / T.sum(denominator*w))
    return qwk

def qwk_cf(predictions, targets, num_classes=5):
    w = np.ones((num_classes,num_classes))
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            w[i,j] = (i-j)**2

    I = np.asarray([25810.0,2443.0,5292.0,873.0,708.0], dtype="float32")
    I = I / np.sum(I)
    #pdb.set_trace()
    numerator = T.dot((targets*I).T, predictions)
    denominator = T.outer((targets*I).sum(axis=0), (predictions*I).sum(axis=0))
    denominator = denominator / T.sum(numerator)
    res = T.sum(numerator*w) / T.sum(denominator*w)
    return res

def get_net_baseline(net_fn, args={}):
    
    X = T.tensor4('X')
    y = T.ivector('y')
    y_ord = T.fmatrix('y_ord')
    y_onehot = T.fmatrix('y_onehot')
    
    cfg = net_fn(args)
    l_out = cfg["l_out"]
    l_in = cfg["l_in"]

    l_exp = DenseLayer(l_out, num_units=1, nonlinearity=linear)
    
    # if we are not learning the 'a' vector, just fix it to something
    if "learn_end" not in args:
        if "hacky_ordinal" in args:
            # just hack something in here for now
            l_exp.W.set_value(np.asarray([[0.],[0.],[0.],[0.]], dtype="float32"))
        else:
            l_exp.W.set_value(np.asarray([[0.],[1.],[2.],[3.],[4.]], dtype="float32"))
    params = get_all_params(l_out, trainable=True)
    if "learn_end" in args:
        assert args["learn_end"] in ["sigm_scaled", "keep_xent"]
        # if we are learning the 'a' vector, make it learnable
        sys.stderr.write("learning W matrix of l_exp\n")
        params += [l_exp.W]
        if args["learn_end"] == "sigm_scaled":
            sys.stderr.write("making l_exp nonlinearity sigm_scaled\n")
            l_exp.nonlinearity = sigm_scaled
        elif args["learn_end"] == "keep_xent":
            sys.stderr.write("optimising both x-ent and squared error\n")
            l_exp.W.set_value(np.asarray([[0.],[1.],[2.],[3.],[4.]], dtype="float32"))

    net_out, exp_out = get_output([l_out,l_exp], {l_in: X})
    net_out_det, exp_out_det = get_output([l_out,l_exp], {l_in: X}, deterministic=True)

    tmp_out = get_output(l_exp, {l_in: X})

    if "hacky_ordinal" in args:
        sys.stderr.write("using binary_crossentropy for both loss and loss_det (hacky ordinal mode)\n")
        loss = binary_crossentropy(net_out, y_ord).mean()
        loss_det = binary_crossentropy(net_out, y_ord).mean()
    elif "learn_end" in args:
        sys.stderr.write("using squared_error for learn_end\n")
        if args["learn_end"]== "keep_xent":
            loss = squared_error(exp_out, y.dimshuffle(0,'x')).mean() + categorical_crossentropy(net_out, y).mean()
        else:
            loss = squared_error(exp_out, y.dimshuffle(0,'x')).mean()
        loss_det = categorical_crossentropy(net_out_det, y).mean()
    elif "qwk" in args:
        sys.stderr.write("using qwk loss\n")
        loss = qwk(net_out, y_onehot)
        loss_det = categorical_crossentropy(net_out_det, y).mean()
    elif "qwk_normw" in args:
        sys.stderr.write("using qwk normw loss\n")
        loss = qwk_normw(net_out, y_onehot)
        loss_det = categorical_crossentropy(net_out_det, y).mean()        
    elif "kappa" in args:
        sys.stderr.write("using kappa (NOT quadratic)\n")
        loss = kappa(net_out, y_onehot)
        loss_det = categorical_crossentropy(net_out_det, y).mean()
    elif "qwk_hybrid" in args:
        sys.stderr.write("using qwk hybrid loss\n")
        loss = categorical_crossentropy(net_out, y).mean() + args["qwk_hybrid"]*qwk(net_out, y_onehot)
        loss_det = categorical_crossentropy(net_out_det, y).mean()
    elif "qwk_cf" in args:
        sys.stderr.write("using qwk cf\n")
        loss = qwk_cf(net_out, y_onehot)
        loss_det = categorical_crossentropy(net_out_det, y).mean()        
    else:
        loss = categorical_crossentropy(net_out, y).mean()
        loss_det = categorical_crossentropy(net_out_det, y).mean()

    # total kappa loss using the expectation trick
    if "klo" in args:
        loss = hp.get_kappa_loss(5)(net_out, y).mean()
        #loss_det = hp.get_kappa_loss(5)(net_out_det, y).mean()
        sys.stderr.write("using kappa loss term\n")
        # for validation set we want to evaluate the
        # cross-entropy still
    
    if "l2" in args:
        sys.stderr.write("adding l2: %f\n" % args["l2"])
        loss += args["l2"]*regularize_network_params(l_out, l2)
        loss_det += args["l2"]*regularize_network_params(l_out, l2)

    #params = get_all_params(l_out, trainable=True)
    sys.stderr.write("params: %s\n" % str(params))
    grads = T.grad(loss, params)
    learning_rate = theano.shared(floatX(0.01)) if "learning_rate" not in args else theano.shared(floatX(args["learning_rate"]))
    sys.stderr.write("learning rate: %f\n" % args["learning_rate"])
    momentum = 0.9 if "momentum" not in args else args["momentum"]
    if "rmsprop" in args:
        sys.stderr.write("using rmsprop instead of nesterov momentum...\n")
        updates = rmsprop(grads, params, learning_rate=learning_rate)
    else:
        updates = nesterov_momentum(grads, params, learning_rate=learning_rate, momentum=momentum)
    # index fns
    bs = args["batch_size"]
    train_fn = theano.function(inputs=[X,y,y_ord,y_onehot], outputs=loss, updates=updates, on_unused_input='warn')
    loss_fn = theano.function(inputs=[X,y,y_ord,y_onehot], outputs=loss_det, on_unused_input='warn')
    
    preds_fn = theano.function(
        inputs=[X],
        outputs=T.argmax(net_out_det,axis=1)
    )
    dist_fn = theano.function(
        inputs=[X],
        outputs=[net_out_det, exp_out_det]
    )
    tmp_fn = theano.function(inputs=[X], outputs=tmp_out)
    
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "preds_fn": preds_fn,
        "dist_fn": dist_fn,
        "tmp_fn":tmp_fn,
        "l_out": l_out,
        "l_exp": l_exp,
        "learning_rate": learning_rate,
        "bs": bs,
        "is_hacky_ordinal": True if "hacky_ordinal" in args else False
    }

def residual_block(layer, n_out_channels, prefix, stride=1, dd={}, nonlinearity=rectify):
    conv = layer
    if stride > 1:
        layer = Pool2DLayer(layer, pool_size=1, stride=stride, mode="average_inc_pad")
    if (n_out_channels != layer.output_shape[1]):
        diff = n_out_channels-layer.output_shape[1]
        if diff % 2 == 0: 
            width_tp = ((diff/2, diff/2),)
        else:
            width_tp = (((diff/2)+1, diff/2),)
        layer = pad(layer, batch_ndim=1, width=width_tp)
    conv = Conv2DLayer(conv, 
                       num_filters=n_out_channels,
                       filter_size=(3,3), 
                       stride=(stride,stride), 
                       pad=(1,1), 
                       nonlinearity=linear, 
                       W=HeNormal(gain="relu") if prefix+"_1" not in dd else dd[prefix+"_1"].W,
                       b=Constant(0.) if prefix+"_1" not in dd else dd[prefix+"_1"].b)
    if prefix+"_1" not in dd:
        dd[prefix+"_1"] = conv
        print prefix+"_1"
    conv = BatchNormLayer(conv,
                          beta=Constant(0.) if prefix+"_bn1" not in dd else dd[prefix+"_bn1"].beta, 
                          gamma=Constant(1.) if prefix+"_bn1" not in dd else dd[prefix+"_bn1"].gamma)
    if prefix+"_bn1" not in dd:
        dd[prefix+"_bn1"] = conv
        print prefix+"_bn1"
    conv = NonlinearityLayer(conv, nonlinearity=nonlinearity)
    conv = Conv2DLayer(conv, 
                       num_filters=n_out_channels,
                       filter_size=(3,3), 
                       stride=(1,1), 
                       pad=(1,1), 
                       nonlinearity=linear, 
                       W=HeNormal(gain="relu") if prefix+"_2" not in dd else dd[prefix+"_2"].W,
                       b=Constant(0.) if prefix+"_2" not in dd else dd[prefix+"_2"].b)
    if prefix+"_2" not in dd:
        dd[prefix+"_2"] = conv
        print prefix+"_2"
    conv = BatchNormLayer(conv,
                          beta=Constant(0.) if prefix+"_bn2" not in dd else dd[prefix+"_bn2"].beta, 
                          gamma=Constant(1.) if prefix+"_bn2" not in dd else dd[prefix+"_bn2"].gamma)
    if prefix+"_bn2" not in dd:
        dd[prefix+"_bn2"] = conv
        print prefix+"_bn2"
    return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity=nonlinearity)


def resnet(quadrant, dd, first_time, nf=[32, 64, 128, 256], N=2, dropout_p=None):
    # 34-layer resnet as per:
    # https://arxiv.org/pdf/1512.03385v1.pdf
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
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)
    layer = residual_block(layer, nf[1], prefix="aa", stride=2, dd=dd)
    for i in range(N):
        layer = residual_block(layer, nf[1], prefix="b%i" % i, dd=dd)
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)
    layer = residual_block(layer, nf[2], prefix="bb%", stride=2, dd=dd)
    for i in range(N):
        layer = residual_block(layer, nf[2], prefix="c%i" % i, dd=dd)
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)
    layer = residual_block(layer, nf[3], prefix="cc", stride=2, dd=dd)
    for i in range(N):
        layer = residual_block(layer, nf[3], prefix="dd%i" % i, dd=dd)
        if dropout_p != None:
            layer = DropoutLayer(layer, p=dropout_p)

    #layer = Pool2DLayer(layer, pool_size=8, stride=1, mode="average_inc_pad")
    # make this dynamic
    layer = Pool2DLayer(layer, pool_size=layer.output_shape[-1], stride=1, mode="average_inc_pad")

    #layer = DenseLayer(layer, 
    #                   num_units=5, 
    #                   nonlinearity=softmax, 
    #                   W=HeNormal() if "softmax" not in dd else dd["softmax"])
    layer = FlattenLayer(layer)
    
    if first_time:
        return dd, layer
    else:
        return layer







def resnet_net_512_baseline(args={}):
    
    l_in = InputLayer( (None, 3, 512, 512) )

    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)

    l_out = DenseLayer(
        conv,
        num_units=5,
        nonlinearity=softmax,
        W=HeNormal()
    )
    
    for layer in get_all_layers(l_out):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    sys.stderr.write("number of params: %i\n" % count_params(l_out))
        
    return {"l_out": l_out,
            "l_in": l_in}


def resnet_net_256_baseline(args={}):
    
    l_in = InputLayer( (None, 3, 256, 256) )

    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)

    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv = DropoutLayer(conv, p=args["end_dropout"])

    l_out = DenseLayer(
        conv,
        num_units=5,
        nonlinearity=softmax,
        W=HeNormal()
    )
    
    for layer in get_all_layers(l_out):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    sys.stderr.write("number of params: %i\n" % count_params(l_out))
        
    return {"l_out": l_out,
            "l_in": l_in}



def resnet_net_224_baseline(args={}):
    
    l_in = InputLayer( (None, 3, 224, 224) )
    
    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)
    
    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv = DropoutLayer(conv, p=args["end_dropout"])
        
    for layer in get_all_layers(conv):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")

    if "out_nonlinearity" in args:
        out_nonlinearity = args["out_nonlinearity"]
        sys.stderr.write("using custom output nonlinearity instead of softmax\n")
    else:
        out_nonlinearity = softmax
    
    l_out = DenseLayer(
        conv,
        num_units=5,
        nonlinearity=out_nonlinearity,
        W=HeNormal()
    )
    
    return {"l_out": l_out,
            "l_in": l_in}




def resnet_net_224_baseline_hacky_ordinal(args={}):
    
    l_in = InputLayer( (None, 3, 224, 224) )
    
    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)
    
    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv = DropoutLayer(conv, p=args["end_dropout"])
        
    for layer in get_all_layers(conv):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    l_out = DenseLayer(
        conv,
        num_units=4,
        nonlinearity=sigmoid,
        W=HeNormal()
    )
    
    return {"l_out": l_out,
            "l_in": l_in}




def resnet_net_224_baseline_ordinal_layer(args={}):
    
    l_in = InputLayer( (None, 3, 224, 224) )
    
    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)
    
    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv = DropoutLayer(conv, p=args["end_dropout"])
        
    for layer in get_all_layers(conv):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")

    #l_pre = DenseLayer(conv, num_units=5-1, nonlinearity=softplus, W=HeNormal())
    #l_uro = UpperRightOnesLayer(l_pre)
    #l_softmax = NonlinearityLayer(l_uro, nonlinearity=sigmoid)
    #l_ord = OrdinalSubtractLayer(l_softmax)
    #l_out = l_ord

    l_out = DenseLayer(conv, num_units=5, nonlinearity=softmax, W=HeNormal())

    for layer in get_all_layers(l_out):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    return {"l_out": l_out,
            "l_in": l_in}



"""
ordinal structure with softplus
"""
def resnet_net_224_baseline_ordinal_layer_v2(args={}):
    
    l_in = InputLayer( (None, 3, 224, 224) )
    
    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)
    
    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv = DropoutLayer(conv, p=args["end_dropout"])
        
    for layer in get_all_layers(conv):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")

    l_pre = DenseLayer(conv, num_units=5, nonlinearity=softplus, W=HeNormal())
    l_uro = UpperRightOnesLayer(l_pre)
    l_div = DivisionLayer(l_uro)
    l_sub = SimpleOrdinalSubtractLayer(l_div)

    l_out = l_sub
    
    #l_out = DenseLayer(conv, num_units=5, nonlinearity=softmax, W=HeNormal())

    for layer in get_all_layers(l_out):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    return {"l_out": l_out,
            "l_in": l_in}



"""
ordinal structure with linear then exp
"""
def resnet_net_224_baseline_ordinal_layer_v3(args={}):
    
    l_in = InputLayer( (None, 3, 224, 224) )
    
    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)
    
    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv = DropoutLayer(conv, p=args["end_dropout"])
        
    for layer in get_all_layers(conv):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")

    l_pre = DenseLayer(conv, num_units=5, nonlinearity=linear, W=HeNormal())
    l_pre = ExpressionLayer(l_pre, lambda X: T.exp(X))
    l_uro = UpperRightOnesLayer(l_pre)
    l_div = DivisionLayer(l_uro)
    l_sub = SimpleOrdinalSubtractLayer(l_div)

    l_out = l_sub
    
    #l_out = DenseLayer(conv, num_units=5, nonlinearity=softmax, W=HeNormal())

    for layer in get_all_layers(l_out):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    return {"l_out": l_out,
            "l_in": l_in}



"""
ordinal structure with tanh then exp
"""
def resnet_net_224_baseline_ordinal_layer_v4(args={}):
    
    l_in = InputLayer( (None, 3, 224, 224) )
    
    dd, conv = resnet(l_in, {}, True, N=args["N"], dropout_p=args["dropout_p"] if "dropout_p" in args else None)
    
    if "end_dropout" in args:
        sys.stderr.write("adding end dropout: %f\n" % args["end_dropout"])
        conv = DropoutLayer(conv, p=args["end_dropout"])
        
    for layer in get_all_layers(conv):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")

    l_pre = DenseLayer(conv, num_units=5, nonlinearity=tanh, W=HeNormal())
    l_pre = ExpressionLayer(l_pre, lambda X: T.exp(X))
    l_uro = UpperRightOnesLayer(l_pre)
    l_div = DivisionLayer(l_uro)
    l_sub = SimpleOrdinalSubtractLayer(l_div)

    l_out = l_sub
    
    #l_out = DenseLayer(conv, num_units=5, nonlinearity=softmax, W=HeNormal())

    for layer in get_all_layers(l_out):
        sys.stderr.write( str(layer) + " " + str(layer.output_shape) + "\n")
    
    return {"l_out": l_out,
            "l_in": l_in}


# ###########################################################################
# DATA LOADING FUNCTIONS
# ###########################################################################

imgen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=359.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.,
    zoom_range=0.02,
    channel_shift_range=0.,
    fill_mode='constant',
    cval=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None)

def load_image_keras(filename, crop):
    img = io.imread(filename)
    img = img_as_float(img)
    img = np.asarray( [ img[...,0], img[...,1], img[...,2] ] ) # reshape
    for i in range(0, img.shape[0]):
        img[i, ...] = (img[i, ...] - np.mean(img[i, ...])) / np.std(img[i,...]) # zmuv
    for xb, _ in imgen.flow( np.asarray([img], dtype=img.dtype), np.asarray([0], dtype="int32")):
        if crop != None:
            img_size = xb[0].shape[-1]
            #print xb[0].shape
            #print img_size
            x_start = np.random.randint(0, img_size-crop+1)
            y_start = np.random.randint(0, img_size-crop+1)
            ret = xb[0][:, y_start:y_start+crop, x_start:x_start+crop]
            #print ret.shape
            return ret
        break
    return xb[0]

def construct_w(cls, num_classes=5):
    w = [0]*num_classes
    for i in range(0, len(w)):
        w[i] = (i-cls)**2
    w = np.asarray(w, dtype="float32")
    #esw = w / 10.0
    return w

# ###############################################################
# DATA ITERATORS
# ###############################################################

def iterate_mnist(X_arr, bs):
    def integer_to_one_hot(X_train):
        total = np.zeros((X_train.shape[0], X_train[0].shape[0], 256)).astype("float32")
        for b in range(0, X_train.shape[0]):
            seqs = np.zeros((X_train[b].shape[0], 256)).astype("float32")
            for i in range(0, X_train[b].shape[0]):
                one_hot = np.zeros((256)).astype("float32")
                one_hot[ X_train[b][i] ] = 1.0
                seqs[i] = one_hot
            total[b] = seqs
        return total
    b = 0
    while True:
        if b*bs >= X_arr.shape[0]:
            break
        this_X_I, this_X = X_arr[b*bs:(b+1)*bs], integer_to_one_hot(X_arr[b*bs:(b+1)*bs])
        yield this_X_I, this_X
        b += 1

def iterate_baseline(X_arr, y_arr, bs, augment, zmuv, crop, DATA_DIR=os.environ["DATA_DIR"]):
    assert X_arr.shape[0] == y_arr.shape[0]
    b = 0
    while True:
        if b*bs >= X_arr.shape[0]:
            break
        this_X, this_y = X_arr[b*bs:(b+1)*bs], y_arr[b*bs:(b+1)*bs]
        images_for_this_X = [ load_image_keras("%s/%s.jpeg" % (DATA_DIR,filename), crop) for filename in this_X ]
        images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
        this_onehot = []
        for elem in this_y:
            one_hot_vector = [0]*5
            one_hot_vector[elem] = 1
            this_onehot.append(one_hot_vector)
        this_onehot = np.asarray(this_onehot, dtype="float32")
        yield images_for_this_X, this_y, this_onehot
        b += 1

def iterate_baseline_balanced(X_arr, y_arr, bs, augment, zmuv, crop, DATA_DIR=os.environ["DATA_DIR"]):
    assert X_arr.shape[0] == y_arr.shape[0]

    num_classes=5
    dd = dict()
    for i in range(0,num_classes):
        dd[i] = []
    for filename, class_idx in zip(X_arr, y_arr):
        dd[class_idx].append(filename)
    iterators = []
    for i in range(0,num_classes):
        iterators.append( cycle(dd[i]) )

    for x in range(0, len(X_arr)//bs):
        this_X = []
        this_y = []
        for i in range(0, len(iterators)):
            this_X += [iterators[i].next() for j in range(bs//num_classes)]
            this_y += [i]*(bs//num_classes)
            #print len(this_X)
        images_for_this_X = [ load_image_keras("%s/%s.jpeg" % (DATA_DIR,filename), crop) for filename in this_X ]
        images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
        this_onehot = []
        for elem in this_y:
            one_hot_vector = [0]*5
            one_hot_vector[elem] = 1
            this_onehot.append(one_hot_vector)
        this_onehot = np.asarray(this_onehot, dtype="float32")
        yield images_for_this_X, this_y, this_onehot
        
def cost(i,j):
    return (i-j)**2

def risk(p_dist, i):
    assert i >= 0 and i < len(p_dist)
    res = 0
    for j in range(0, len(p_dist)):
        res += p_dist[j]*cost(i,j)
    return res

def ord_encode(k, num_classes=5):
    """
    The ordinal encoding scheme proposed in:
    https://arxiv.org/pdf/0704.1028.pdf
    """
    arr = [0]*(num_classes-1)
    if k == 0:
        return arr
    for i in range(0, k):
        arr[i] = 1
    return arr

def convert_hacky_ord_dist_into_label(arr, num_classes=5):
    mask = arr >= 0.5
    if False not in mask:
        return num_classes-1
    else:
        return np.where(mask == False)[0][0]

    
def init_exp(out_file, header, resume=None, out_layer=None, resume_legacy=False):
    """
    Initialises an experiment by creating an output file and optionally
    loading the model's parameters.
    """
    f = None
    if resume == None:
        if out_file != None:
            f = open("%s.txt" % out_file, "wb")
            f.write(",".join(header) + "\n")
            print ",".join(header)
    else:
        sys.stderr.write("init_exp(): resuming training...\n")
        if out_file != None:
            f = open("%s.txt" % out_file, "ab")
        with open(resume) as g:
            if resume_legacy:
                set_all_param_values(out_layer.input_layer, pickle.load(g))
            else:
                set_all_param_values(out_layer, pickle.load(g))
    return f

def epoch_iterator(x1, num_epochs):
    """
    Basic epoch shuffle iterator
    """
    idxs = [x for x in range(0, x1.shape[0])]
    for epoch in range(0, num_epochs):
        np.random.shuffle(idxs)
        x1 = x1[idxs]
        #x2 = x2[idxs]
        yield x1
                
def train_baseline(net_cfg,
          num_epochs,
          data,
          out_file=None,
          debug=False,
          resume=None,
          resume_legacy=False,
          augment=True,
          zmuv=False,
          crop=None,
          balanced_minibatches=False,
          schedule={},
          skip_train=False,
          save_valid_dists=False):
    # prepare the out_file
    l_out, l_exp = net_cfg["l_out"], net_cfg["l_exp"]

    header = ["epoch","train_loss","avg_valid_loss","valid_accuracy","valid_kappa","valid_kappa_exp","valid_kappa_mc","time"]
    f = init_exp(out_file=out_file, header=header, resume=resume, out_layer=l_exp, resume_legacy=resume_legacy)
            
    # extract functions
    X_train, y_train, X_valid, y_valid = data
    train_fn, loss_fn, preds_fn, dist_fn = net_cfg["train_fn"], net_cfg["loss_fn"], net_cfg["preds_fn"], net_cfg["dist_fn"]
    learning_rate = net_cfg["learning_rate"]
    bs = net_cfg["bs"]
    is_hacky_ordinal = net_cfg["is_hacky_ordinal"]
    
    # training
    for epoch, (X_train, y_train) in enumerate(epoch_iterator(X_train,y_train)):
        
        if epoch+1 in schedule:
            sys.stderr.write("train_baseline(): changing learning rate to: %f\n" % schedule[epoch+1])
            learning_rate.set_value( floatX(schedule[epoch+1]) )

        if balanced_minibatches:
            iterator_function = iterate_baseline_balanced
            sys.stderr.write("train_baseline(): using balanced minibatch iterator\n")
        else:
            iterator_function = iterate_baseline

        if debug:
            pdb.set_trace()

        if not skip_train:

            # training loop
            this_train_losses = []
            t0 = time()
            for X_train_batch, y_train_batch, y_onehot_train_batch in iterator_function(X_train, y_train, bs, augment, zmuv, crop):
                y_ord_train_batch = np.asarray([ord_encode(elem) for elem in y_train_batch], dtype="float32")
                this_train_losses.append( train_fn(X_train_batch, y_train_batch, y_ord_train_batch, y_onehot_train_batch) )
            time_taken = time() - t0        

            # validation loss loop
            this_valid_losses = []
            for X_valid_batch, y_valid_batch, y_onehot_valid_batch in iterate_baseline(X_valid, y_valid, bs, False, zmuv, crop):
                y_ord_valid_batch = np.asarray([ord_encode(elem) for elem in y_valid_batch], dtype="float32")
                this_valid_losses.append( loss_fn(X_valid_batch, y_valid_batch, y_ord_valid_batch, y_onehot_valid_batch) )
            avg_valid_loss = np.mean(this_valid_losses)
        
        # validation accuracy loop
        valid_preds = []
        valid_preds_exp = []
        valid_preds_metacost = []
        valid_preds_hacky_ordinal = []

        if save_valid_dists != None:
            svd = open("%s.txt" % save_valid_dists, "wb")
            
        for X_valid_batch, y_valid_batch, _ in iterate_baseline(X_valid, y_valid, bs, False, zmuv, crop):

            # just do argmax to get the predictions for valid_kappa, but only if not is_hacky_ordinal

            preds = preds_fn(X_valid_batch).tolist()
            valid_preds += preds
            
            dists, dists_exp = dist_fn(X_valid_batch)

            if is_hacky_ordinal:
                for dist in dists:
                    valid_preds_hacky_ordinal.append( convert_hacky_ord_dist_into_label(dist) )

            #pdb.set_trace()
            
            # compute the valid kappa exp
            # this metric makes no sense for the hacky ordinal setting
            for dist in dists_exp:
                valid_preds_exp.append( min(int(np.round(dist)), 4) )

            # compute the metacost valid kappa
            # this metric makes no sense for the hacky ordinal setting
            for dist in dists:
                valid_preds_metacost.append( np.argmin([ risk(dist, i) for i in range(0, len(dist)) ]) )

            # optionally save the valid dists to file
            if save_valid_dists != None:
                for z in range(0, len(dists)):
                    svd.write( ",".join( [str(d) for d in dists[z]] ) + "," + str(y_valid_batch[z])  + "\n")
        if save_valid_dists != None:
            svd.close()
            
        valid_acc = np.mean(valid_preds == y_valid)

        if not is_hacky_ordinal:        
            valid_kappa = hp.weighted_kappa(human_rater=valid_preds, actual_rater=y_valid, num_classes=5)
        else:
            valid_kappa = hp.weighted_kappa(human_rater=valid_preds_hacky_ordinal, actual_rater=y_valid, num_classes=5)
            
        valid_kappa_exp = hp.weighted_kappa(human_rater=valid_preds_exp, actual_rater=y_valid, num_classes=5)
        valid_kappa_mc = hp.weighted_kappa(human_rater=valid_preds_metacost, actual_rater=y_valid, num_classes=5)
        
        ## ------------ ##
        if f != None:
            f.write(
                "%i,%f,%f,%f,%f,%f,%f,%f\n" %
                    (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, valid_kappa, valid_kappa_exp, valid_kappa_mc, time_taken) 
            )
            f.flush()
        if print_out:
            print "%i,%f,%f,%f,%f,%f,%f,%f" % (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, valid_kappa, valid_kappa_exp, valid_kappa_mc, time_taken)
            
        with open("models/%s.modelv2.%i" % (os.path.basename(out_file),epoch+1), "wb") as g:
            pickle.dump(get_all_param_values(l_exp), g, pickle.HIGHEST_PROTOCOL) 
            
    if f != None:
        f.close()


# ######################################################################################################
# NESTED LSTM EXPERIMENTS
# ######################################################################################################

def lstm(args):
    num_units = args["num_units"] 
    seqlen, num_classes = 28*28, 256
    l_in = InputLayer((None, seqlen, num_classes))
    l_lstm = LSTMLayer(l_in, num_units=num_units, precompute_input=True, unroll_scan=True)
    l_shp = ReshapeLayer(l_lstm, (-1, num_units))
    l_dense = DenseLayer(l_shp, num_units=num_classes, nonlinearity=softmax)
    l_shp2 = ReshapeLayer(l_dense, (-1, seqlen, num_classes))
    for layer in get_all_layers(l_shp2):
        print layer, layer.output_shape
    print "num params:", count_params(l_shp2)
    return l_shp2

def lstm_orthog(args):

    gate_parameters = Gate(W_in=Orthogonal(), W_hid=Orthogonal(), b=Constant(0.))
    cell_parameters = Gate(W_in=Orthogonal(), W_hid=Orthogonal(), W_cell=None, b=Constant(0.), nonlinearity=tanh)
    
    num_units = args["num_units"] 
    seqlen, num_classes = 28*28, 256
    l_in = InputLayer((None, seqlen, num_classes))
    l_lstm = LSTMLayer(
        l_in,
        num_units=num_units,
        precompute_input=True,
        ingate=gate_parameters,
        outgate=gate_parameters,
        forgetgate=gate_parameters,
        cell=cell_parameters
    )
    l_shp = ReshapeLayer(l_lstm, (-1, num_units))
    l_dense = DenseLayer(l_shp, num_units=num_classes, nonlinearity=softmax)
    l_shp2 = ReshapeLayer(l_dense, (-1, seqlen, num_classes))
    for layer in get_all_layers(l_shp2):
        print layer, layer.output_shape
    print "num params:", count_params(l_shp2)
    return l_shp2


def get_net_lstm(net_cfg, args):
    l_out = net_cfg(args)
    num_classes = l_out.output_shape[-1]
    
    l_exp_reshp = ReshapeLayer(l_out, (-1, l_out.input_layer.output_shape[1]))
    l_exp_dense = DenseLayer(l_exp_reshp, num_units=1, nonlinearity=linear)
    l_exp_dense.W.set_value( np.asarray([ [i] for i in range(0, num_classes) ], dtype="float32") )
    l_exp = ReshapeLayer(l_exp_dense, (-1, l_out.output_shape[1], 1))
    
    X = T.tensor3('X')
    X_I = T.imatrix('y')
    net_out, exp_out = get_output([l_out, l_exp], X)
    params = get_all_params(l_out, trainable=True)
    if "klo" not in args:
        print "get_net_lstm(): using cross-entropy"
        loss = categorical_crossentropy(net_out[:,0:-1,:], X[:,1::,:]).mean()
        loss_det = loss
    else:
        print "get_net_lstm(): using 'klo' loss"
        loss = squared_error( exp_out[:,0:-1,0], X_I[:,1::]).mean()
        loss_det = categorical_crossentropy(net_out[:,0:-1,:], X[:,1::,:]).mean()
        #params += [l_exp_dense.W]

    learning_rate = theano.shared(floatX(0.01)) if "learning_rate" not in args else theano.shared(floatX(args["learning_rate"]))
    sys.stderr.write("learning rate: %f\n" % args["learning_rate"])
    grads = T.grad(loss, params)
    momentum = 0.9 if "momentum" not in args else args["momentum"]
    if "rmsprop" in args:
        sys.stderr.write("get_net_lstm(): using rmsprop\n")
        updates = rmsprop(grads, params, learning_rate=learning_rate)
    else:
        updates = nesterov_momentum(grads, params, learning_rate=learning_rate, momentum=momentum)

    train_fn = theano.function([X, X_I], loss, updates=updates, on_unused_input='warn')
    loss_fn = theano.function([X, X_I], loss_det, on_unused_input='warn')
    dist_fn = theano.function(
        inputs=[X],
        outputs=[net_out, exp_out]
    )
    
    return {
        "train_fn":train_fn,
        "loss_fn":loss_fn,
        "dist_fn":dist_fn,
        "bs": args["batch_size"],
        "l_out": l_out,
        "l_exp": l_exp,
        "learning_rate":learning_rate
    }

def train_lstm(net_cfg,
          num_epochs,
          data,
          out_file=None,
          debug=False,
          resume=None,
          resume_legacy=False,
          save_every=5,
          schedule={}):

    # prepare the out_file
    l_out, l_exp = net_cfg["l_out"], net_cfg["l_exp"]

    header = ["epoch","train_loss","avg_valid_loss","valid_accuracy","valid_kappa","valid_kappa_exp","time"]
    formats = ["%i", "%f", "%f", "%f", "%f", "%f", "%f"]
    f = init_exp(out_file=out_file, header=header, resume=resume, out_layer=l_exp, resume_legacy=resume_legacy)
            
    # extract functions
    X_train_I, X_valid_I = data
    X_valid_I_flatten = X_valid_I.flatten()
    train_fn, loss_fn, dist_fn = net_cfg["train_fn"], net_cfg["loss_fn"], net_cfg["dist_fn"]
    learning_rate = net_cfg["learning_rate"]
    bs = net_cfg["bs"]

    #pdb.set_trace()
    
    # training
    for epoch, (X_train_I) in enumerate(epoch_iterator(X_train_I,num_epochs)):
        
        if epoch+1 in schedule:
            sys.stderr.write("train_baseline(): changing learning rate to: %f\n" % schedule[epoch+1])
            learning_rate.set_value( floatX(schedule[epoch+1]) )

        iterator_function = iterate_mnist

        #if debug:
        #    pdb.set_trace()

        # training loop
        this_train_losses = []
        t0 = time()
        for X_train_I_batch, X_train_batch in iterator_function(X_train_I, bs):
            this_train_losses.append( train_fn(X_train_batch, X_train_I_batch) )
            if debug:
                break
        time_taken = time() - t0

        # validation loss loop
        this_valid_losses = []
        for X_valid_I_batch, X_valid_batch in iterator_function(X_valid_I, bs):
            this_valid_losses.append( loss_fn(X_valid_batch, X_valid_I_batch) )
            if debug:
                break
        avg_valid_loss = np.mean(this_valid_losses)
        
        valid_preds = [] # argmax predictions
        valid_preds_exp = [] # exp predictions
            
        for X_valid_I_batch, X_valid_batch in iterator_function(X_valid_I, bs):

            #pdb.set_trace()

            dists, dists_exp = dist_fn(X_valid_batch)

            # compute the argmax
            # (bs, 784, 256) --> (bs, 784) --> (bs*784)
            this_argmaxes = np.argmax(dists, axis=2).flatten()
            valid_preds += this_argmaxes.tolist()

            # compute the expectation
            # (bs, 784, 1) --> (bs, 784) --> (bs*784)
            this_exp = dists_exp[:,:,0].flatten()
            valid_preds_exp += np.round(this_exp).astype("uint8").tolist()

        valid_acc = (X_valid_I_flatten==valid_preds).mean()
        valid_kappa = hp.weighted_kappa(human_rater=valid_preds, actual_rater=X_valid_I_flatten, num_classes=256)
        valid_kappa_exp = hp.weighted_kappa(human_rater=valid_preds_exp, actual_rater=X_valid_I_flatten, num_classes=256)
        
        #pdb.set_trace()
            
        ## ------------ ##
        if f != None:
            f.write(
                (",".join(formats) + "\n") %
                    (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, valid_kappa, valid_kappa_exp, time_taken) 
            )
            f.flush()
        print (",".join(formats)) % (epoch+1, np.mean(this_train_losses), avg_valid_loss, valid_acc, valid_kappa, valid_kappa_exp, time_taken)

        #if epoch % save_every == 0:
        with open("models_lstm/%s.modelv2.%i" % (os.path.basename(out_file),epoch+1), "wb") as g:
            pickle.dump(get_all_param_values(l_exp), g, pickle.HIGHEST_PROTOCOL)
            
    if f != None:
        f.close()

