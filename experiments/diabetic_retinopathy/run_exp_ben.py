import experiment
import sys
import os
import random
random.seed(0)
sys.path.append("../../modules/")
from helper import *

args = dict()
args["max_epochs"] = 100
args["alpha"] = 0.01
args["seed"] = 100 # not 0!!!
args["batch_size"] = 128

filenames, labels = load_dr_train_labels( os.environ["DATA_DIR"] + os.path.sep + "trainLabels.csv" )
args["filenames"] = filenames
args["prefix"] = os.environ["DATA_DIR"] + os.path.sep + "train-trim-ben-256"
labels = np.asarray(labels, dtype="int32")
args["X_train"] = np.asarray([x for x in range(0, len(filenames))], dtype="int32")
args["y_train"] = labels

args["zmuv"] = False
args["augment"] = False

"""
part1
"""
args["out_model"] = "ilya_net_ben.model"
args["out_stats"] = "ilya_net_ben"
experiment.train(args)
