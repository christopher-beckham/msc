"""
Original code at: https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb
Original authors: ebenolson, webeng
"""

import lasagne
import numpy as np
import pickle
import scipy
import theano
import theano.tensor as T
from lasagne.utils import floatX
from skimage.io import imread, imsave
from skimage import filters
import train_ae
from time import time
from lasagne.layers import *
import sys
sys.path.append("../../modules/")
import matplotlib.pyplot as plt
import imp
import argparse

def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g

def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()

# -----

# defaults
"""
CONFIG_NAME = "vgg_a_subset.py"
MODEL_NAME = "output/vgg_a_subset.2.pkl"
NPY_FILE = "train_data_minimal.npy"
REF_IMAGE_INDEX = 5
STYLE_COEF = 1e7
VARIATION_COEF = 0.001
NUM_IMAGES = 1
NUM_ITERS = 6
OUT_FOLDER = "output_neat"
LAYERS = "conv4_2,conv1_1,conv2_1,conv3_1,conv4_1,conv5_1".split(",")
"""

parser = argparse.ArgumentParser(description="style transfer")
parser.add_argument("--config_name", dest="config_name", help="name of model config file")
parser.add_argument("--model_name", dest="model_name", help="name of model file")
parser.add_argument("--npy_file", dest="npy_file", help="name of npy train file (to get reference image)")
parser.add_argument("--ref_image_index", type=int, dest="ref_image_index", help="index of ref image")
parser.add_argument("--style_coef", dest="style_coef", type=float, help="style coef")
parser.add_argument("--variation_coef", dest="variation_coef", type=float, help="variation coef")
parser.add_argument("--num_images", dest="num_images", type=int, help="number of images to generate")
parser.add_argument("--num_iters", dest="num_iters", type=int, help="number of l-bfgs iters")
parser.add_argument("--out_folder", dest="out_folder", help="variation coef")
args = parser.parse_args()

CONFIG_NAME = args.config_name
MODEL_NAME = args.model_name
NPY_FILE = args.npy_file
REF_IMAGE_INDEX = args.ref_image_index
STYLE_COEF = args.style_coef
VARIATION_COEF = args.variation_coef
NUM_IMAGES = args.num_images
NUM_ITERS = args.num_iters
OUT_FOLDER = args.out_folder

if NUM_ITERS < 6:
    print "error: num iters must be >= 6!"
    sys.exit(1)

# -----

config = imp.load_source("cfg", CONFIG_NAME)
net_raw = config.get_net({})

with open(MODEL_NAME) as f:
    dat = pickle.load(f)
    try:
        set_all_param_values(net_raw["l_out"], dat)
    except Exception as e:
        set_all_param_values(net_raw["l_out"], dat["param values"])
layers = net_raw["target_layers"]
print "target layers:", layers

heightmap = np.load(NPY_FILE)[REF_IMAGE_INDEX:REF_IMAGE_INDEX+1]
photo = heightmap

# -----

input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
art_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                for k, output in zip(layers.keys(), outputs)}
generated_image = theano.shared(floatX(np.random.uniform(0, 1, photo.shape)))
gen_features = lasagne.layers.get_output(layers.values(), generated_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

losses = []
# style loss
for key in layers:
    losses.append(STYLE_COEF*style_loss(art_features, gen_features, key))
# total variation loss
losses.append(VARIATION_COEF * total_variation_loss(generated_image))
total_loss = sum(losses)

grad = T.grad(total_loss, generated_image)

# -----

# Theano functions to evaluate loss and gradient
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)

# Helper functions to interface with scipy.optimize
def eval_loss(x0):
    x0 = floatX(x0.reshape(photo.shape))
    generated_image.set_value(x0)
    return f_loss().astype('float64')

def eval_grad(x0):
    x0 = floatX(x0.reshape(photo.shape))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype('float64')

for iter_ in range(0, NUM_IMAGES):
    print "image #: %i" % (iter_+1)
    t0 = time()
    generated_image.set_value(floatX(np.random.uniform(0, 1, photo.shape)))
    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)
    for i in range(0, NUM_ITERS):
        print "  iter #: %i" % (i+1)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
        x0 = generated_image.get_value().astype('float64')
        xs.append(x0)
    # generate grid image
    cc = 1
    nrow, ncol = 2, 3
    for i in range(0, nrow):
        for j in range(0, ncol):
            plt.subplot(nrow, ncol, cc)
            plt.imshow(xs[cc-1][0][0], cmap="gray")
            cc += 1
    plt.savefig(OUT_FOLDER + "/%i_evolution.png" % t0)
    # save final heightmap
    imsave(out_folder + "/%i.png" % t0, arr=xs[-1][0][0])

