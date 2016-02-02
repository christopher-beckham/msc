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
args["seed"] = 0
args["batch_size"] = 128

filenames, labels = load_dr_train_labels( os.environ["DATA_DIR"] + os.path.sep + "trainLabels.csv" )
args["filenames"] = filenames
args["prefix"] = os.environ["DATA_DIR"] + os.path.sep + "train-trim-256"
labels = np.asarray(labels, dtype="int32")
args["X_train"] = np.asarray([x for x in range(0, len(filenames))], dtype="int32")
args["y_train"] = labels

"""
part1
"""
#args["out_model"] = "ilya_net.model"
#args["out_stats"] = "ilya_net"
#experiment.train(args)

"""
part2
"""
#args["out_model"] = "ilya_net.2.model"
#args["out_stats"] = "ilya_net.2"
#args["in_model"] = "ilya_net.model"
#experiment.train(args)

"""
part3
"""

args["out_model"] = "ilya_net.3.model"
args["out_stats"] = "ilya_net.3"
args["in_model"] = "ilya_net.2.model"
experiment.train(args)
