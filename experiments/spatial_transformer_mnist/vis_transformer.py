import experiment
import sys
sys.path.append("../../modules/")
from helper import *
# ---
import lasagne
from lasagne.layers import *
import theano
from theano import tensor as T
# ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_name = "exp1.model"
args = dict()
# because we used cudnn on cuda, we have to use conv2d/maxpool2d here, 
# which means changing those classes' flip_filter behaviours
args["dont_flip_filters"] = False

args["input_shape"] = (None, 1, 60, 60)
args["max_epochs"] = 100
args["alpha"] = 0.01
args["seed"] = 0
args["batch_size"] = 128
args["out_model"] = model_name
args["out_stats"] = "exp1"
net = experiment.get_net(args)
net.load_params_from(model_name)
#net.initialize()
print net

layers = net.get_all_layers()
l_trans = None
for layer in layers:
    if isinstance(layer, TransformerLayer):
        l_trans = layer

X = T.tensor4('x')
get_trans_out = theano.function([X], lasagne.layers.get_output(l_trans, X))

train_set = load_cluttered_mnist_train_only("../../data/mnist_cluttered_60x60_6distortions.npz")
Xt, _ = train_set

import scipy

for idx in range(0, 100):
    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #ax1.imshow(Xt[idx:idx+1][0][0], interpolation='nearest', vmin=0, vmax=1, cmap='Greys')
    #ax2.imshow(get_trans_out(Xt[idx:idx+1])[0][0], interpolation='nearest', vmin=0, vmax=1, cmap='Greys')
    #f.savefig("transforms/%i.png" % idx)
    scipy.misc.toimage(get_trans_out(Xt[idx:idx+1])[0][0], cmin=0, cmax=1).save("transforms/%i.png" % idx)
