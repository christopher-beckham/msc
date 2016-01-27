import experiment
import sys
sys.path.append("../../modules/")
from helper import *

#data = load_mnist("../../data/mnist.pkl.gz")
train_set = load_cluttered_mnist_train_only("../../data/mnist_cluttered_60x60_6distortions.npz")
args = dict()
args["input_shape"] = (None, 1, 60, 60)
args["X_train"] = train_set[0]
args["y_train"] = train_set[1]
args["max_epochs"] = 100
args["alpha"] = 0.01
args["seed"] = 0
args["batch_size"] = 128
args["out_model"] = "exp1.model"
args["out_stats"] = "exp1"
net = experiment.train(args)
