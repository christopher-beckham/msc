import sys
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
import os
sys.path.append(os.environ["EARTH_MOVER"])
from keras.preprocessing.image import ImageDataGenerator
# --
from adience_test import load_xval_data, load_pre_split_data

def load_pre_split_data_in_memory(dirname):
    xt, yt, xv, yv = load_pre_split_data(dirname)

    arr=[]
    for i in range(0, xt.shape[0]):
        arr.append(iterators._load_image_from_filename(xt[i]))
        if i % 5000 == 0:
            print i
            pdb.set_trace()
    
    #xt = [ iterators._load_image_from_filename(filename) for filename in xt ]
    #print xt.shape

if __name__ == '__main__':
    # testing
    from skimage.io import imread
    import pdb

    #xt, yt, xv, yv = load_xval_data(0, "/data/lisatmp4/beckhamc/adience_data/aligned_256x256/")

    load_pre_split_data_in_memory("/data/lisatmp4/beckhamc/adience_data/aligned_256x256/")
    
    """
    imgen = ImageDataGenerator()
    for elem in xt:
        #img = imread(elem)
        img = iterators._load_image_from_filename(elem,imgen=imgen,crop=224,debug=True)
        #print elem, img.shape
        #assert ( img.shape == (3,224,224) or img.shape == (224,224) )
    """
