from skimage.io import imread
from skimage import img_as_float
import numpy as np
import pdb
import helpers

def _augment_image(img, imgen=None, crop=None):
    if imgen == None and crop == None:
        return img
    for xb, _ in imgen.flow( np.asarray([img], dtype=img.dtype), np.asarray([0], dtype="int32") ):
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

def _load_image_from_filename(filename, imgen=None, crop=None, debug=False):
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
    if debug:
        print "initial: ", filename, img.shape
    # if the image is black and white
    if len(img.shape) == 2:
        img = np.asarray([img,img,img])
    else:
        img = np.asarray( [ img[...,0], img[...,1], img[...,2] ] ) # reshape
    for i in range(0, img.shape[0]):
        img[i, ...] = (img[i, ...] - np.mean(img[i, ...])) / np.std(img[i,...]) # zmuv
    if imgen == None:
        return img
    if debug:
        print "post-process: ", filename, img.shape
    return _augment_image(img, imgen=imgen, crop=crop)

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

def _shuffle(X_arr, y_arr, rnd_state):
    if rnd_state != None:
        idxs = [x for x in range(X_arr.shape[0])]
        rnd_state.shuffle(idxs)
        X_arr, y_arr = X_arr[idxs], y_arr[idxs]
    return X_arr, y_arr

def _get_slices(length, bs):
    slices = []
    b = 0
    while True:
        if b*bs >= length:
            break
        slices.append( slice(b*bs, (b+1)*bs) )
        b += 1
    return slices

#TODO: refactor since it shares code with iterate_filenames
def iterate_images(imgen=None, crop=None):
    def _iterate_images(X_arr, y_arr, bs, num_classes):
        """
        :param X_arr: images
        :param y_arr:
        :return:
        """
        assert X_arr.shape[0] == y_arr.shape[0]
        X_arr, y_arr = _shuffle(X_arr, y_arr, rnd_state)
        b = 0
        while True:
            if b*bs >= X_arr.shape[0]:
                break
            this_X, this_y = X_arr[b*bs:(b+1)*bs], y_arr[b*bs:(b+1)*bs]
            images_for_this_X = [ _augment_image(img, imgen, crop) for img in this_X ]
            images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
            this_onehot = helpers.label_to_one_hot(this_y, num_classes)
            yield images_for_this_X, this_onehot
            b += 1        
    return _iterate_images

def iterate_filenames(imgen=None, crop=None):
    """
    return a filename iterator, conditioned on a keras
    data augmentor and a crop size
    :param imgen:
    :param crop:
    :return:
    """
    def _iterate_filenames(X_arr, y_arr, bs, num_classes, rnd_state=None):
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
        X_arr, y_arr = _shuffle(X_arr, y_arr, rnd_state)
        b = 0
        while True:
            if b*bs >= X_arr.shape[0]:
                break
            this_X, this_y = X_arr[b*bs:(b+1)*bs], y_arr[b*bs:(b+1)*bs]
            images_for_this_X = [ _load_image_from_filename(filename, imgen, crop) for filename in this_X ]
            images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
            this_onehot = helpers.label_to_one_hot(this_y, num_classes)
            yield images_for_this_X, this_onehot
            b += 1
    return _iterate_filenames

def iterate_hdf5(imgen=None, crop=None):
    def _iterate_hdf5(X_arr, y_arr, bs, num_classes, rnd_state=None):
        assert X_arr.shape[0] == y_arr.shape[0]
        slices = _get_slices(X_arr.shape[0], bs)
        if rnd_state != None:
            rnd_state.shuffle(slices)
        for elem in slices:
            this_X, this_y = X_arr[elem], y_arr[elem]
            images_for_this_X = [ _augment_image(img, imgen, crop) for img in this_X ]
            images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
            this_onehot = helpers.label_to_one_hot(this_y, num_classes)
            yield images_for_this_X, this_onehot
    return _iterate_hdf5

if __name__ == '__main__':
    pass
