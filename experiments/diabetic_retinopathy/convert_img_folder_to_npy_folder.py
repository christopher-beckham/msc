from skimage import io
from skimage import img_as_float
import numpy as np
import glob
import sys
import os

src_folder = sys.argv[1]
dest_folder = sys.argv[2]

files = glob.glob("%s/*.jpeg" % src_folder)
for filename in files:
    print filename
    img = io.imread(filename)
    img = img_as_float(img)
    dest = "%s/%s.npy" % (dest_folder, os.path.basename(filename).replace(".jpeg",""))
    np.save(dest,img)
