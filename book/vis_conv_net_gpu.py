import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.layers.dnn import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.regularization import *
from lasagne.random import get_rng
from lasagne.updates import *
from lasagne.init import *
import numpy as np
import sys
sys.path.append("../modules/")
import helper as hp
from lasagne.utils import floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import cPickle as pickle
from theano.tensor import TensorType
from theano.ifelse import ifelse
from time import time
from scipy import stats
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import io
import skimage.transform
import urllib

class KeepOneFeatureMap(Layer):
    def __init__(self, incoming, which_ones, **kwargs):
        super(KeepOneFeatureMap, self).__init__(incoming, **kwargs)
        self.which_ones = which_ones
        self.incoming = incoming
    def get_output_for(self, input, **kwargs):
        mask = T.zeros(input.shape)
        mask = theano.tensor.set_subtensor(mask[:,self.which_ones,:,:], 1.0)
        return mask*input

def print_net(l):
    for layer in get_all_layers(l):
        print layer, layer.output_shape
    print count_params(l)

def image_net(which_one):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = Conv2DDNNLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
    net['norm1'] = LocalResponseNormalization2DLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = MaxPool2DLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = Conv2DDNNLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = Conv2DDNNLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv4'] = Conv2DDNNLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv5'] = Conv2DDNNLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=softmax)
    output_layer = net['fc8']
    # conv5
    return {"l_conv5": net["conv5"], "l_out": output_layer}

l = T.ivector('l')
cfg = image_net(l)
l_conv5 = cfg["l_conv5"]
l_keep = KeepOneFeatureMap(l_conv5, which_ones=l)
# ok, now do the inverse layers
l_decode = l_keep
for layer in get_all_layers(l_keep.input_layer)[::-1]:
    if isinstance(layer, InputLayer):
        break
    l_decode = InverseLayer(l_decode, layer)
print_net(l_decode)

# make sure this is run:
# wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl

with open("vgg_cnn_s.pkl") as f:
    imagenet_model = pickle.load(f)
    CLASSES = imagenet_model['synset words']
    MEAN_IMAGE = imagenet_model['mean image']
    set_all_param_values(l_conv5, imagenet_model["values"][0:10])

def prep_image(url):
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    return rawim, floatX(im[np.newaxis])

rawim, im = prep_image("http://static.flickr.com/1372/582719105_4e4016397e.jpg")


X = T.tensor4('X')
y = T.ivector('y')
net_out = get_output(l_decode, X, deterministic=True)
out_fn = theano.function([X, l], net_out)


mode = "normal"
num_fms = 512

if mode == "normal":
    for i in range(num_fms):
        #grid[i].imshow(out_fn(im, i)[0][0], cmap="gray")
        decoded_img = out_fn( im, np.asarray([i], dtype="int32") )[0]
        #print decoded_img.shape
        # ok, the decoded image is in the shape (3, 224, 224)
        # so let's make it (224, 224, 3) so we can save the image

        decoded_img = decoded_img + MEAN_IMAGE
        new_img = np.zeros((224,224,3), dtype="uint8")
        for x in range(0, 224):
            for y in range(0, 224):
                vec = np.asarray([ decoded_img[2,x,y], decoded_img[1,x,y], decoded_img[0,x,y] ], dtype="uint8")
                new_img[x,y] = vec        

        plt.imsave("output/%i.png" % i, new_img, cmap="gray")
elif mode == "random":
    # visualise random subset of fm's
    window_size=1
    how_many=100
    for iter_ in range(how_many):
        idxs = [k for k in range(0, num_fms)]
        np.random.shuffle(idxs)
        idxs = idxs[0:window_size]
        idxs = np.asarray(idxs, dtype="int32")
        decoded_img = out_fn(im, idxs)[0]
        #print decoded_img.shape
        # ok, the decoded image is in the shape (3, 224, 224)
        # so let's make it (224, 224, 3) so we can save the image

        decoded_img = decoded_img + MEAN_IMAGE
        new_img = np.zeros((224,224,3), dtype="uint8")
        for x in range(0, 224):
            for y in range(0, 224):
                vec = np.asarray([ decoded_img[2,x,y], decoded_img[1,x,y], decoded_img[0,x,y] ], dtype="uint8")
                new_img[x,y] = vec

        
        """
        new_img = np.zeros((224,224,3))
        for x in range(0, 224):
            for y in range(0, 224):
                new_img[x,y,0] = decoded_img[0,x,y]
                new_img[x,y,1] = decoded_img[1,x,y]
                new_img[x,y,2] = decoded_img[2,x,y]
        """
        #new_img = decoded_img.transpose(1,2,0)
        
        print new_img.shape
        plt.imsave("output/%i.png" % iter_, new_img)
        # test
        #print out_fn(im, idxs)[0].shape
