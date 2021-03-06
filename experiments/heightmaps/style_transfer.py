"""
Original code at: https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb
Original authors: ebenolson, webeng
"""

import os
import matplotlib
if os.environ["HOSTNAME"] == "cuda4.rdgi.polymtl.ca":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lasagne
import numpy as np
import pickle
import scipy
import theano
import theano.tensor as T
from lasagne.utils import floatX
from skimage.io import imread, imsave
from skimage import filters
from skimage.filters import gaussian_filter
import train_ae
from time import time
from lasagne.layers import *
import sys
sys.path.append("../../modules/")
import os
import imp
import argparse

#plt.subplot(2,1,1)
#plt.plot([1,2,3])
#plt.savefig("/tmp/stuff.png")

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
parser.add_argument("--config_name", dest="config_name", help="name of model config file", required=True)
parser.add_argument("--model_name", dest="model_name", help="name of model file", required=True)
parser.add_argument("--npy_file", dest="npy_file", help="name of npy train file (to get reference image)", required=True)
parser.add_argument("--ref_image_index", type=int, dest="ref_image_index", help="index of ref image", required=True)
parser.add_argument("--reference_coef", dest="reference_coef", type=float, help="reference coef (wip)")
parser.add_argument("--style_coef", dest="style_coef", type=float, help="style coef", required=True)
parser.add_argument("--variation_coef", dest="variation_coef", type=float, help="variation coef", required=True)
parser.add_argument("--num_images", dest="num_images", type=int, help="number of images to generate", required=True)
parser.add_argument("--num_iters", dest="num_iters", type=int, help="number of l-bfgs iters", required=True)
#parser.add_argument("--out_folder", dest="out_folder", help="variation coef")
parser.add_argument("--grid", dest="grid", help="plot a grid?")
parser.add_argument("--sigma", dest="sigma", help="sigma for if we gauss blur the white noise image")
parser.add_argument("--outfile", dest="outfile", help="out file", required=True)
parser.add_argument("--cheat_index", dest="cheat_index", type=int, help="use another image in the training set instead of white noise")
args = parser.parse_args()

CONFIG_NAME = args.config_name
MODEL_NAME = args.model_name
NPY_FILE = args.npy_file
REF_IMAGE_INDEX = args.ref_image_index
STYLE_COEF = args.style_coef
VARIATION_COEF = args.variation_coef
NUM_IMAGES = args.num_images
NUM_ITERS = args.num_iters
#OUT_FOLDER = args.out_folder

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
if args.reference_coef != None or args.reference_coef != 0:
    sys.stderr.write("args.reference_coef > 0, adding input layer to list...\n")
    layers["input"] = get_all_layers(net_raw["l_out"])[0]
print "target layers:", layers

data = np.load(NPY_FILE)

heightmap = data[REF_IMAGE_INDEX:REF_IMAGE_INDEX+1]
if net_raw["use_rgb"]:
    # we have to make this (1,3,256,256) instead of (1,1,256,256)
    heightmap = np.asarray( [ [ heightmap[0,0,:,:], heightmap[0,0,:,:], heightmap[0,0,:,:] ] ] )
    print heightmap.shape
photo = heightmap

# -----

input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
art_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                for k, output in zip(layers.keys(), outputs)}
generated_image = theano.shared( np.random.uniform(0, 1, photo.shape).astype( theano.config.floatX  ) )
gen_features = lasagne.layers.get_output(layers.values(), generated_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

losses = []
# style loss
for key in layers:
    if key == "input":
        losses.append(args.reference_coef*style_loss(art_features, gen_features, key))
    else: 
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
    return f_loss().astype( theano.config.floatX )

def eval_grad(x0):
    x0 = floatX(x0.reshape(photo.shape))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype( theano.config.floatX )

for iter_ in range(0, NUM_IMAGES):
    print "image #: %i" % (iter_+1)
    t0 = time()
    white_noise = np.random.uniform(0, 1, photo.shape).astype(  theano.config.floatX  )
    if args.sigma > 0:
        white_noise = gaussian_filter(white_noise[0][0], sigma=args.sigma).reshape(photo.shape)
        sys.stderr.write("adding gaussian noise\n")
    elif args.cheat_index != None:
        white_noise = data[args.cheat_index : args.cheat_index+1]
        sys.stderr.write("using image index %i as the seed image" % args.cheat_index)
    generated_image.set_value(white_noise)
    x0 = generated_image.get_value().astype( theano.config.floatX )
    xs = []
    xs.append(x0)
    for i in range(0, NUM_ITERS):
        print "  iter #: %i" % (i+1)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
        x0 = generated_image.get_value().astype( theano.config.floatX )
        # scale between 0 and 1, grrr
        x0 = (x0 - np.min(x0)) / (np.max(x0) - np.min(x0))
        xs.append(x0)
    # generate grid image
    if args.grid != "no":
        cc = 1
        nrow, ncol = 2, 3
        for i in range(0, nrow):
            for j in range(0, ncol):
                plt.subplot(nrow, ncol, cc)
                plt.imshow(xs[cc-1][0][0], cmap="gray")
                cc += 1
        plt.savefig("%s_evolution.png" % args.outfile)
    # save final heightmap
    imsave("%s.png" % args.outfile, arr=xs[-1][0][0])

