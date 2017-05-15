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
    all_X, all_y = load_all_data(dataset_dir)
    rnd = np.random.RandomState(0)
    idxs = [x for x in range(len(all_y))]
    rnd.shuffle(idxs)
    train_idxs = idxs[0:int(split_pt*len(idxs))]
    valid_idxs = idxs[int(split_pt*len(idxs))::]
    return all_X[train_idxs], all_y[train_idxs], all_X[valid_idxs], all_y[valid_idxs]

def _write_pre_split_data_to_hdf5(data_dir, split_pt, out_file):
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
