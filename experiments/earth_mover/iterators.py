from skimage.io import imread
from skimage import img_as_float
import numpy as np

def _load_image_from_filename(filename, imgen=None, crop=None):
    """
    load an image from a filename, optionally specifiying a keras
    data augmentor and/or a random crop size
    :param filename:
    :param imgen:
    :param crop:
    :return:
    """
    img = imread(filename)
    img = img_as_float(img)
    img = np.asarray( [ img[...,0], img[...,1], img[...,2] ] ) # reshape
    for i in range(0, img.shape[0]):
        img[i, ...] = (img[i, ...] - np.mean(img[i, ...])) / np.std(img[i,...]) # zmuv
    if imgen == None:
        return img
    for xb, _ in imgen.flow( np.asarray([img], dtype=img.dtype), np.asarray([0], dtype="int32")):
        if crop != None:
            img_size = xb[0].shape[-1]
            #print xb[0].shape
            #print img_size
            ret = xb[0]
            if crop != None:
                x_start = np.random.randint(0, img_size-crop+1)
                y_start = np.random.randint(0, img_size-crop+1)
                ret = xb[0][:, y_start:y_start+crop, x_start:x_start+crop]
            return ret
        break
    return xb[0]

def iterate(X_arr, y_arr, bs, **args):
    """
    basic iterator
    :param X_arr:
    :param y_arr:
    :param bs:
    :return:
    """
    assert X_arr.shape[0] == y_arr.shape[0]
    b = 0
    while True:
        if b*bs >= X_arr.shape[0]:
            break
        this_X, this_y = X_arr[b*bs:(b+1)*bs], y_arr[b*bs:(b+1)*bs]
        yield this_X, this_y
        b += 1

def iterate_filenames(X_arr, y_arr, bs, num_classes, imgen=None, crop=None):
    """
    load an image from a list of filenames, optionally specifying a keras
    data augmentor and/or a random crop size
    return x,y, where x = image tensor and y = one-hot matrix
    :param X_arr: filenames
    :param y_arr: class labels
    :param bs: batch size
    :param num_classes: num of classes
    :param imgen: keras data augmentor
    :param crop: optional crop size
    :return:
    """
    assert X_arr.shape[0] == y_arr.shape[0]
    b = 0
    while True:
        if b*bs >= X_arr.shape[0]:
            break
        this_X, this_y = X_arr[b*bs:(b+1)*bs], y_arr[b*bs:(b+1)*bs]
        images_for_this_X = [ _load_image_from_filename(filename, imgen, crop) for filename in this_X ]
        images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
        this_onehot = []
        for elem in this_y:
            one_hot_vector = [0]*num_classes
            one_hot_vector[elem] = 1
            this_onehot.append(one_hot_vector)
        this_onehot = np.asarray(this_onehot, dtype="float32")
        yield images_for_this_X, this_onehot
        b += 1