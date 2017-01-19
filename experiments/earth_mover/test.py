import datasets.adience
from iterators import iterate_filenames
import os
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
from keras.preprocessing.image import ImageDataGenerator

imgen = ImageDataGenerator()
xt, yt, xv, yv = datasets.adience.get_fold(0)
for xb, yb in iterate_filenames(xt, yt, bs=32, imgen=imgen, num_classes=8, crop=None):
    print xb, yb
    break

#print datasets.adience.load_image(xt[0], 224).shape

# ---------