import sys
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
import os
sys.path.append(os.environ["EARTH_MOVER"])
import iterators
from keras.preprocessing.image import ImageDataGenerator
# --
#from adience_test import load_xval_data, load_pre_split_data
import numpy as np
import common

def _read_fold(fold_idx, dataset_dir, debug=False):
    ages = dict() # debug
    dd = dict() # map column names to indices
    classes = {
        "(0, 2)": 0, "(4, 6)": 1, "(8, 12)": 2, "(15, 20)": 3, "(25, 32)": 4, "(38, 43)": 5, "(48, 53)": 6, "(60, 100)": 7
    }
    delim = "\t"
    # -----
    X, y = [], []
    with open("%s/../fold_%i_data.txt" % (dataset_dir, fold_idx)) as f:
        header = f.readline().rstrip().split(delim)
        for k in range(0, len(header)):
            dd[ header[k] ] = k
        for line in f:
            line = line.rstrip().split(delim)
            #print line
            #print line[dd["user_id"]], line[dd["original_image"]], line[dd["face_id"]], line[dd["age"]]
            img_loc = "%s/%s/landmark_aligned_face.%s.%s" % \
                (dataset_dir, line[dd["user_id"]], line[dd["face_id"]], line[dd["original_image"]])
            if line[dd["age"]] in classes:
                #print img_loc, classes[ line[dd["age"]] ]
                X.append(img_loc)
                y.append(classes[ line[dd["age"]] ])          
            if line[dd["age"]] not in ages:
                ages[ line[dd["age"]] ] = 0
            ages[ line[dd["age"]] ] += 1
    if debug:
        print ages
    return X, y

def load_xval_data(valid_fold_idx, dataset_dir):
    """
    No longer used.
    """
    assert 0 <= valid_fold_idx <= 3
    tot_folds = ["fold_0_data.txt", "fold_1_data.txt", "fold_2_data.txt", "fold_3_data.txt"]
    training_folds_idxs = [ i for i in range(0, len(tot_folds)) if i != valid_fold_idx ]
    #valid_fold = tot_folds[valid_fold_idx]
    training_X = []
    training_y = []
    for training_fold_idx in training_folds_idxs:
        x, y = _read_fold(training_fold_idx, dataset_dir)
        training_X += x
        training_y += y
    valid_X, valid_y = _read_fold(valid_fold_idx, dataset_dir)
    training_X = np.asarray(training_X)
    valid_X = np.asarray(valid_X)
    training_y = np.asarray(training_y, dtype="int32")
    valid_y = np.asarray(valid_y, dtype="int32")
    return training_X, training_y, valid_X, valid_y

def load_all_data(dataset_dir):
    """
    Load all training + valid data.
    This does not return images, just filenames.
    """
    training_folds_idxs = [ i for i in range(0, 4) ]
    #valid_fold = tot_folds[valid_fold_idx]
    training_X = []
    training_y = []
    for training_fold_idx in training_folds_idxs:
        x, y = _read_fold(training_fold_idx, dataset_dir)
        training_X += x
        training_y += y
    training_X = np.asarray(training_X)
    training_y = np.asarray(training_y, dtype="int32")
    return training_X, training_y

def load_pre_split_data(dataset_dir, split_pt=0.9):
    """
    Load all training + valid data, shuffle the data,
    and randomly split into train and valid sets with a
    proportion defined by `split_pt`.
    This does not return images, just filenames.
    """
    all_X, all_y = load_all_data(dataset_dir)
    rnd = np.random.RandomState(0)
    idxs = [x for x in range(len(all_y))]
    rnd.shuffle(idxs)
    train_idxs = idxs[0:int(split_pt*len(idxs))]
    valid_idxs = idxs[int(split_pt*len(idxs))::]
    return all_X[train_idxs], all_y[train_idxs], all_X[valid_idxs], all_y[valid_idxs]

def _write_pre_split_data_to_hdf5(data_dir, split_pt, out_file):
    """
    Write training and validation images out to HDF5.
    """
    import h5py
    xt_filenames, yt, xv_filenames, yv = load_pre_split_data(data_dir, split_pt)
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

def load_test_data(dataset_dir):
    """
    Load only test data.
    This does not return images, just filenames.
    """
    # folds 0..3 are train/valid, so fold 4 is test
    X_test, y_test = _read_fold(4, dataset_dir)
    return np.asarray(X_test), np.asarray(y_test,dtype="int32")

def _write_pre_split_test_data_to_hdf5(data_dir, out_file):
    """
    Write test images out to HDF5.
    """
    import h5py
    xtest_filenames, ytest = load_test_data(data_dir)
    h5f = h5py.File(out_file, 'w')
    h5f.create_dataset('xtest' , shape=(len(xtest_filenames),3,256,256), dtype="float32")
    h5f.create_dataset('ytest' , shape=(len(ytest),), dtype="int32")
    for i in range(0, xtest_filenames.shape[0]):
        h5f['xtest'][i] = iterators._load_image_from_filename(xtest_filenames[i])
        h5f['ytest'][i] = ytest[i]
    h5f.close()
    
def create_imbalanced_dataset_from_h5(h5_filename, out_filename, classes_to_reduce, reduce_by):
    """
    Reduce the training set of an h5 by reducing certain classes by a certain percentage.
    For example, if we have 4 classes [0,1,2,3], and we wish to reduce class 0 and 1 by 80%,
    `classes_to_reduce = [0,1]` and `reduce_by = 0.8`.
    """
    # NOTE: requires loading whole dataset into RAM
    import h5py
    f = h5py.File(h5_filename, "r")
    xt, yt, xv, yv = f['xt'], f['yt'], f['xv'], f['yv']
    idxs_by_class = common.group_indices_by_class(yt[:])
    xt_m, yt_m = common.create_imbalanced_data(xt[:], yt[:], idxs_by_class, classes_to_reduce, reduce_by)
    g = h5py.File(out_filename,'w')
    g.create_dataset("xt", data=xt_m)
    g.create_dataset("yt", data=yt_m)
    g.create_dataset("xv", data=xv)
    g.create_dataset("yv", data=yv)
    g.flush()
    g.close()
    f.close()
    
if __name__ == '__main__':
    # testing
    from skimage.io import imread
    import pdb

    #xt, yt, xv, yv = load_xval_data(0, "/data/lisatmp4/beckhamc/adience_data/aligned_256x256/")

    #load_pre_split_data_in_memory("/data/lisatmp4/beckhamc/adience_data/aligned_256x256/")

    # 0.9 split
    #_write_pre_split_data_to_hdf5("/data/lisatmp4/beckhamc/adience_data/aligned_256x256/", "/data/lisatmp4/beckhamc/hdf5/adience_256.h5")

    # 0.5 split
    #_write_pre_split_data_to_hdf5("/data/lisatmp4/beckhamc/adience_data/aligned_256x256/", 0.5, "/data/lisatmp4/beckhamc/hdf5/adience_256_50-50.h5")

    # 0.25 split
    #_write_pre_split_data_to_hdf5("/data/lisatmp4/beckhamc/adience_data/aligned_256x256/", 0.25, "/data/lisatmp4/beckhamc/hdf5/adience_256_25-75.h5")

    # 0.10 split
    #_write_pre_split_data_to_hdf5("/data/lisatmp4/beckhamc/adience_data/aligned_256x256/", 0.1, "/data/lisatmp4/beckhamc/hdf5/adience_256_10-90.h5")

    # create imbalanced data where every class other than 4 is reduced to 5% (0.05) size
    #create_imbalanced_dataset_from_h5(
    #    h5_filename="/data/lisatmp4/beckhamc/hdf5/adience_256.h5",
    #    out_filename="/data/lisatmp4/beckhamc/hdf5/adience_256_0.05re4.h5",
    #    classes_to_reduce=[0,1,2,3,5,6,7],
    #    reduce_by=0.05
    #)

    #create_imbalanced_dataset_from_h5(
    #    h5_filename="/data/lisatmp4/beckhamc/hdf5/adience_256.h5",
    #    out_filename="/data/lisatmp4/beckhamc/hdf5/adience_256_0.1re4.h5",
    #    classes_to_reduce=[0,1,2,3,5,6,7],
    #    reduce_by=0.1
    #)

    """
    create_imbalanced_dataset_from_h5(
        h5_filename="/data/lisatmp4/beckhamc/hdf5/adience_256.h5",
        out_filename="/data/lisatmp4/beckhamc/hdf5/adience_256_0.2re4.h5",
        classes_to_reduce=[0,1,2,3,5,6,7],
        reduce_by=0.2
    )
    """

    """
    create_imbalanced_dataset_from_h5(
        h5_filename="/data/lisatmp4/beckhamc/hdf5/adience_256.h5",
        out_filename="/data/lisatmp4/beckhamc/hdf5/adience_256_0.3re4.h5",
        classes_to_reduce=[0,1,2,3,5,6,7],
        reduce_by=0.3
    )
    """
 
    #common.group_indices_by_class
    
    """
    # yt statistics for 90-10 split:
    >>> (yt==0).sum()
    1818
    >>> (yt==1).sum()
    1409
    >>> (yt==2).sum()
    1610
    >>> (yt==3).sum()
    1288
    >>> (yt==4).sum()
    3552
    >>> (yt==5).sum()
    1587
    >>> (yt==6).sum()
    524
    >>> (yt==7).sum()
    557
    """
    
    # intentionally imbalance the class??
    ##
    
    """
    imgen = ImageDataGenerator()
    for elem in xt:
        #img = imread(elem)
        img = iterators._load_image_from_filename(elem,imgen=imgen,crop=224,debug=True)
        #print elem, img.shape
        #assert ( img.shape == (3,224,224) or img.shape == (224,224) )
    """

    _write_pre_split_test_data_to_hdf5("/data/lisatmp4/beckhamc/adience_data/aligned_256x256/", "/data/lisatmp4/beckhamc/hdf5/adience_test_256.h5")
