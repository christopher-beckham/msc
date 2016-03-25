import sys
import os
from skimage.io import imread, imsave
import glob
import scipy.misc

data_folder = os.environ["DATA_DIR"] + "/clean"
out_folder = os.environ["DATA_DIR"] + "/train"

def get_chunks(img, offset=20, crop_size=256):
    chunks = []
    for y in range(0, img.shape[0] - crop_size+1, offset):
        for x in range(0, img.shape[1]-crop_size+1, offset):
            chunk = img[ y:y+crop_size, x:x+crop_size ]
            if chunk.shape != (crop_size, crop_size):
                continue
            #print chunk.shape
            chunks.append(chunk)
    return chunks

i = 0
for filename in glob.glob(data_folder + "/*.png"):
    print filename
    chunks = get_chunks( imread(filename) )
    for chunk in chunks:
        #imsave(out_folder + ("/%i.png" % i), chunk )
        scipy.misc.imsave(out_folder + ("/%i.png" % i), chunk )
        i += 1