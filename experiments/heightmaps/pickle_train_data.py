import sys
import os
from skimage.io import imread, imsave
from skimage import img_as_float
import glob
import scipy.misc
import cPickle as pickle
import numpy as np

data_folder = os.environ["DATA_DIR"] + "/clean"
out_folder = os.environ["DATA_DIR"] + "/train"

data = []
for filename in glob.glob(out_folder + "/*.png"):
    print filename
    data.append( img_as_float(imread(filename)).reshape( (1,256,256) ) )
data = np.asarray(data, dtype="float32")
#with open(out_folder + "/train_data.pkl", "wb") as f:
#    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
np.save(out_folder + "/train_data.npy", data)
