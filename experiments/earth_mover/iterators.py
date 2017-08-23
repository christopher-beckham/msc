from skimage.io import imread
from skimage import img_as_float
import numpy as np
import pdb
import helpers
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import MultiProcessing
from fuel.streams import DataStream, ServerDataStream
import eugene.io as eugene_io

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

# TODO: look at https://github.com/fchollet/keras/issues/3338
def augment_images(imgs, imgen=None, crop=None):
    new_shape = imgs.shape
    if crop != None:
        new_shape = (imgs.shape[0], imgs.shape[1], crop, crop)
    new_imgs = np.zeros(new_shape)
    for i in range(0, new_imgs.shape[0]):
        new_imgs[i] = _augment_image(imgs[i], imgen, crop)
    return new_imgs.astype("float32")

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

def iterate_semi_shuffle_async(imgen=None, crop=None):
    def _iterate_semi_shuffle_async(X_arr, y_arr, bs, num_classes, rnd_state=None):
        def preprocessor_functor(batch):
            batch_X, batch_y = batch
            return augment_images(batch_X, imgen, crop), batch_y
        assert X_arr.shape[0] == y_arr.shape[0]
        shuffle = True
        if rnd_state == None:
            shuffle = False
        df = eugene_io.data_flow(data=[X_arr, y_arr], batch_size=bs, loop_forever=False, shuffle=shuffle, rnd_state=rnd_state, preprocessor=preprocessor_functor)
        for this_X, this_y in df.flow():
            yield this_X, helpers.label_to_one_hot(this_y, num_classes)
    return _iterate_semi_shuffle_async

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

def iterate_hdf5_fuel(dataset_dir, imgen=None, crop=None):
    def _iterate_hdf5_fuel(X_id, y_arr, bs, num_classes, rnd_state=None):
        # HACKY: because this is meant to be used in tandem with
        # the load_pre_split_data_into_memory_as_hdf5_fuel method,
        # X_arr is not a tensor, it is a string
        assert isinstance(X_arr, str)
        dataset = H5PYDataset(dataset_dir, which_sets=(X_id, ))
        iterator_scheme = SequentialScheme(batch_size=bs, examples=dataset.num_examples)
        ds = MultiProcessing(DataStream(dataset=dataset, iteration_scheme=iterator_scheme), max_store=20)
        for this_X, this_y in ds.get_epoch_iterator():
            images_for_this_X = [ _augment_image(img, imgen, crop) for img in this_X ]
            images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
            this_onehot = helpers.label_to_one_hot(this_y, num_classes)
            yield images_for_this_X, this_onehot
    return _iterate_hdf5_fuel

def iterate_hdf5_fuel_server(imgen=None, crop=None):
    def _iterate_hdf5_fuel_server(X_port, y_arr, num_classes, **kwargs):
        # HACKY: because this is meant to be used in tandem with
        # the load_pre_split_data_into_memory_as_hdf5_fuel_server method,
        # X_id is not a tensor, it is a SERVER PORT
        assert isinstance(X_port, int )
        ds = ServerDataStream((X_port,), produces_examples=False, port=port)
        for this_X, this_y in ds.get_epoch_iterator():
            images_for_this_X = [ _augment_image(img, imgen, crop) for img in this_X ]
            images_for_this_X = np.asarray(images_for_this_X, dtype="float32")
            this_onehot = helpers.label_to_one_hot(this_y, num_classes)
            yield images_for_this_X, this_onehot
    return _iterate_hdf5_fuel_server

if __name__ == '__main__':
    # test fuel server
    #itr = iterate_hdf5_fuel_server()
    #for x, y in itr("train", None, num_classes=5):
    #    print x.shape, y.shape
    #    break

    #X = np.eye(10).astype("float32")
    #y = np.argmax(X,axis=1).astype("int32")
    
    #for Xb, yb in iterate_semi_shuffle_async()(X, y, bs=2, num_classes=1, rnd_state=np.random.RandomState(10)):
    #    print Xb, yb

    import datasets.dr
    from keras.preprocessing.image import ImageDataGenerator
    Xt, yt, Xv, yv = datasets.dr.load_pre_split_data_into_memory_as_hdf5("/data/lisatmp4/beckhamc/hdf5/dr.h5")
    itr = iterate_hdf5(ImageDataGenerator(), 224)
    rnd_state = np.random.RandomState(0)
    for Xb, yb in itr(Xv, yv, 128, 5, rnd_state=rnd_state):
        print np.argmax(yb,axis=1)
        break
    print yv[0:100]
    for Xb, yb in itr(Xv, yv, 128, 5, rnd_state=rnd_state):
        print np.argmax(yb,axis=1)
        break
    print yv[0:100]
