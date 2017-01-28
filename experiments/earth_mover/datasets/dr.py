import numpy as np
import cPickle as pickle
import sys
import os
sys.path.append(os.environ["EARTH_MOVER"])
import iterators
from keras.preprocessing.image import ImageDataGenerator
import pdb

def load_pre_split_data(out_dir):
    """
    this returns 10% validation, not 5% like in the
    squared error reformulation paper!!!
    """
    
    with open("%s/datasets/dr.pkl" % os.environ["EARTH_MOVER"]) as f:
        dat = pickle.load(f)
    X_left, X_right, y_left, y_right = dat
    X_left = [ ("%s/%s.jpeg" % (out_dir, elem)) for elem in X_left ]
    X_right = [ ("%s/%s.jpeg" % (out_dir, elem)) for elem in X_right ]
    
    X_left = np.asarray(X_left)
    X_right = np.asarray(X_right)
    y_left = np.asarray(y_left, dtype="int32")
    y_right = np.asarray(y_right, dtype="int32")
    #np.random.seed(0)
    rnd = np.random.RandomState(0)
    idxs = [x for x in range(0, len(X_left))]
    #np.random.shuffle(idxs)
    rnd.shuffle(idxs)

    t=0.9

    X_train_left = X_left[idxs][0 : int(t*X_left.shape[0])]
    X_train_right = X_right[idxs][0 : int(t*X_right.shape[0])]
    y_train_left = y_left[idxs][0 : int(t*y_left.shape[0])]
    y_train_right = y_right[idxs][0 : int(t*y_left.shape[0])]
    X_valid_left = X_left[idxs][int(t*X_left.shape[0]) ::]
    X_valid_right = X_right[idxs][int(t*X_right.shape[0]) ::]
    y_valid_left = y_left[idxs][int(t*y_left.shape[0]) ::]
    y_valid_right = y_right[idxs][int(t*y_right.shape[0]) ::]
    # ok, fix now
    X_train = np.hstack((X_train_left, X_train_right))
    X_valid = np.hstack((X_valid_left,X_valid_right))
    y_train = np.hstack((y_train_left,y_train_right))
    y_valid = np.hstack((y_valid_left,y_valid_right))

    return X_train, y_train, X_valid, y_valid

def load_pre_split_data_in_memory(data_dir):
    xt_filenames, yt, xv_filenames, yv = load_pre_split_data(data_dir)
    xt = np.zeros((len(xt_filenames),3,256,256),dtype="float32")
    xv = np.zeros((len(xv_filenames),3,256,256),dtype="float32")
    for i in range(0, xt_filenames.shape[0]):
        xt[i] = iterators._load_image_from_filename(xt_filenames[i])
    for i in range(0, xv_filenames.shape[0]):
        xv[i] = iterators._load_image_from_filename(xv_filenames[i])
    return xt, yt, xv, yv

def _write_pre_split_data_to_hdf5(data_dir, out_file):
    import h5py
    xt_filenames, yt, xv_filenames, yv = load_pre_split_data(data_dir)
    h5f = h5py.File(out_file, 'w')
    h5f.create_dataset('xt' , shape=(len(xt_filenames),3,256,256), dtype="float32")
    h5f.create_dataset('yt' , shape=(len(yt),), dtype="int32")
    for i in range(0, xt_filenames.shape[0]):
        h5f['xt'][i] = iterators._load_image_from_filename(xt_filenames[i])
        h5f['yt'][i] = yt[i]
    h5f.create_dataset('xv' , shape=(len(xv_filenames),3,256,256), dtype="float32")
    h5f.create_dataset('yv' , shape=(len(yv),), dtype="int32")
    for i in range(0, xv_filenames.shape[0]):
        h5f['xv'][i] = iterators._load_image_from_filename(xv_filenames[i])
        h5f['yv'][i] = yv[i]
    h5f.close()

def _write_pre_split_data_to_hdf5_fuel(data_dir, out_file, debug=False):
    from fuel.datasets.hdf5 import H5PYDataset
    import h5py
    xt_filenames, yt, xv_filenames, yv = load_pre_split_data(data_dir)
    if debug:
        xt_filenames = xt_filenames[0:128]
        yt = yt[0:128]
        xv_filenames = xv_filenames[0:128]
        yv = yv[0:128]
    f = h5py.File(out_file, mode='w')
    image_features = f.create_dataset('image_features', (xt_filenames.shape[0]+xv_filenames.shape[0], 3, 256, 256), dtype='float32')
    targets = f.create_dataset('targets', (xt_filenames.shape[0]+xv_filenames.shape[0],), dtype='int32')
    for i in range(0, xt_filenames.shape[0]):
        f['image_features'][i] = iterators._load_image_from_filename(xt_filenames[i])
        f['targets'][i] = yt[i]
    for j in range(0, xv_filenames.shape[0]):
        f['image_features'][i+1+j] = iterators._load_image_from_filename(xv_filenames[i])
        f['targets'][i] = yv[i]
    split_dict = {
        'train': {
            'image_features': (0, xt_filenames.shape[0]),
            'targets': (0, xt_filenames.shape[0])},
        'test': {
            'image_features': (xt_filenames.shape[0], xt_filenames.shape[0]+xv_filenames.shape[0]),
            'targets': (xt_filenames.shape[0], xt_filenames.shape[0]+xv_filenames.shape[0])}
        }
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

def _test_timing():
    from time import time
    import sys
    import h5py
    #dataset_dir="/tmp/train-trim-ben-256/"
    dataset_dir="/data/lisatmp4/beckhamc/train-trim-ben-256/"
    xt, yt, xv, yv = load_pre_split_data(dataset_dir)
    imgen = ImageDataGenerator(horizontal_flip=True)
    # do some timing experiments
    # loading images from disk (incl augmentation)
    t0 = time()
    for elem in xt[0:128]:
        img = iterators._load_image_from_filename(elem,imgen=imgen,crop=224,debug=False)
    print "time taken to load 128 images from disk: %f" % (time()-t0)
    # loading images into memory first, and then returning augmented
    xbatch = []
    for elem in xt[0:128]:
        xbatch.append(iterators._load_image_from_filename(elem, imgen=None,crop=None))
    xbatch = np.asarray(xbatch)
    t0 = time()
    for xt, _ in iterators.iterate_images()(xbatch, yt[0:128], 128, 5):
        print xt.shape
    print "time taken to augment 128 images from memory: %f" % (time()-t0)
    # loading images from hdf5 and returning augmented
    h5_dir = "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    h5f = h5py.File(h5_dir, 'r')
    t0 = time()
    tmp = [ iterators._augment_image(img, imgen, 224) for img in h5f['xt'][0:128] ]
    print "time taken to augment 128 images from h5: %f" % (time()-t0)
    # loading images from shuffled hdf5 and returning augmented
    idxs = [x for x in range(0, h5f['xt'].shape[0])]
    np.random.shuffle(idxs)
    tmp0 = np.asarray([ h5f['xt'][i] for i in idxs[0:128] ], dtype=h5f['xt'].dtype)
    tmp = [ iterators._augment_image(img, imgen, 224) for img in tmp0 ]
    print "time taken to augment 128 images from shuffled h5: %f" % (time()-t0)

def load_pre_split_data_into_memory_as_hdf5(dataset_dir):
    import h5py
    h5f = h5py.File(dataset_dir, 'r')
    return h5f['xt'], h5f['yt'], h5f['xv'], h5f['yv']
    
def _test_load_from_memory():
    imgen = ImageDataGenerator(horizontal_flip=True)
    xt, yt, xv, yv = load_pre_split_data_in_memory("/data/lisatmp4/beckhamc/train-trim-ben-256/")
    it = iterators.iterate_images(imgen=imgen,crop=224)
    for xt, yt in it(xt, yt, bs=128, num_classes=5):
        print xt.shape, yt.shape

def _test_hdf5_load():
    xt, yt, xv, yv = load_pre_split_data_into_memory_as_hdf5("/data/lisatmp4/beckhamc/hdf5/dr.h5")
    pdb.set_trace()
    imgen = ImageDataGenerator(horizontal_flip=True)
    for xt, yt in iterators.iterate_hdf5(imgen=imgen, crop=224)(xt, yt, bs=128, num_classes=5, rnd_state=np.random.RandomState(0)):
        print xt.shape, yt.shape
        
#def _test_hdf5_iterator():
    #iterate_hdf5
        
if __name__ == '__main__':
    #_test_timing()
    #_test_load_from_memory()
    #_test_h5()
    #_write_pre_split_data_to_hdf5("/data/lisatmp4/beckhamc/train-trim-ben-256/", "/data/lisatmp4/beckhamc/hdf5/dr.h5")
    #_test_hdf5_load()
    _write_pre_split_data_to_hdf5_fuel("/data/lisatmp4/beckhamc/train-trim-ben-256/", "/data/lisatmp4/beckhamc/hdf5/dr_fuel_test.h5", debug=True)
    
    
