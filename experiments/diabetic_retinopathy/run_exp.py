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
args["out_model"] = "ilya_net.model"
args["out_stats"] = "ilya_net"
filenames, labels = load_dr_train_labels( os.environ["DATA_DIR"] + os.path.sep + "trainLabels.csv" )
args["filenames"] = filenames
args["prefix"] = os.environ["DATA_DIR"] + os.path.sep + "train-trim-256"
labels = np.asarray(labels, dtype="int32")
args["X_train"] = np.asarray([x for x in range(0, len(filenames))], dtype="int32")
args["y_train"] = labels

experiment.train(args)


"""
it = ImageBatchIterator(filenames=filenames, prefix=os.environ["DATA_DIR"]+os.path.sep+"train-trim-256", zmuv=True, batch_size=10)
it(np.asarray([x for x in range(0, len(filenames))], dtype="int32"), labels)
for img in it:
    print img
"""