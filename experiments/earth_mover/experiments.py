import sys
import os
import architectures
import architectures.simple
import architectures.resnet
import datasets
import datasets.mnist
#import datasets.adience
import datasets.dr
import iterators
from base import NeuralNet
import numpy as np
import lasagne
from lasagne.updates import *
from lasagne.utils import *
from lasagne.nonlinearities import *
import theano
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil

def copy_if_not_exist(dest_file, from_file):
    dirname = os.path.dirname(dest_file)
    if not os.path.isfile(dest_file):
        print "copying data from %s to %s" % (from_file, dest_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy(src=from_file, dst=dest_file)

#####################
# MNIST EXPERIMENTS #
#####################

def mnist_baseline():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.simple.light_net, num_classes=10, learning_rate=0.01, momentum=0.9, mode="x_ent", args={}, debug=True)
    nn.train(
        data=datasets.mnist.load_mnist("../../data/mnist.pkl.gz"),
        iterator_fn=iterators.iterate,
        batch_size=32,
        num_epochs=10,
        out_dir="output/mnist_baseline",
        save_to="models/mnist_baseline",
        debug=False
    )

# EMD isn't appropriate for this since it's not an ordinal problem
def mnist_earth_mover():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.simple.light_net, num_classes=10, learning_rate=0.01, momentum=0.9, mode="emd2", args={}, debug=True)
    nn.train(
        data=datasets.mnist.load_mnist("../../data/mnist.pkl.gz"),
        iterator_fn=iterators.iterate,
        batch_size=32,
        num_epochs=50,
        schedule={25: 0.001},
        out_dir="output/mnist_earth_mover",
        debug=False
    )

# TODO: problematic??
def mnist_exp():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.simple.light_net, num_classes=10, learning_rate=0.001, momentum=0.9, mode="exp", args={})
    nn.train(
        data=datasets.mnist.load_mnist("../../data/mnist.pkl.gz"),
        iterator_fn=iterators.iterate,
        batch_size=128,
        num_epochs=100,
        out_dir="output/mnist_exp",
        debug=False
    )

#######################
# ADIENCE EXPERIMENTS #
#######################

def adience_baseline_f0():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.01, momentum=0.9, mode="x_ent",
                   args={}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_xval_data(0, "/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_baseline_f0", save_to="models/adience_baseline_f0", debug=False)

def adience_baseline_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.01, momentum=0.9, mode="x_ent",
                   args={}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    #nn.train(
    #    data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
    #    iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/adience_baseline_pre_split", save_to="models/adience_baseline_pre_split", debug=False)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_baseline_pre_split", save_to="models/adience_baseline_pre_split",
        resume="models/adience_baseline_pre_split.modelv1.14.bak", debug=False)
    
def adience_emd2_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.01, momentum=0.9, mode="emd2",
                   args={}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_emd2_pre_split", save_to="models/adience_emd2_pre_split", debug=False)
    
def adience_exp_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.01, momentum=0.9, mode="exp",
                   args={}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_exp_pre_split", save_to="models/adience_exp_pre_split", debug=False)

def adience_exp_l2_1e4_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.01, momentum=0.9, mode="exp",
                   args={"l2":1e-4}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_exp_l2-1e-4_pre_split", save_to="models/adience_exp_l2-1e-4_pre_split", debug=False)
    
def adience_exp_lr01_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.1, momentum=0.9, mode="exp",
                   args={}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_exp_lr0.1_pre_split", save_to="models/adience_exp_lr0.1_pre_split", debug=False)

def adience_exp_l2_1e4_lr01_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.1, momentum=0.9, mode="exp",
                   args={"l2":1e-4}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_exp_l2-1e-4_lr0.1_pre_split", save_to="models/adience_exp_l2-1e-4_lr0.1_pre_split", debug=False)

# adam experiments
    
def adience_exp_l2_1e4_adam_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="exp",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_exp_l2-1e-4_adam_pre_split", save_to="models/adience_exp_l2-1e-4_adam_pre_split", debug=False)

def adience_xent_l2_1e4_adam_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="x_ent",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_xent_l2-1e-4_adam_pre_split", save_to="models/adience_xent_l2-1e-4_adam_pre_split", debug=False)

def adience_xent_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="x_ent",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)),"momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/adience_256.h5", "/data/lisatmp4/beckhamc/hdf5/adience_256.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/adience_xent_l2-1e-4_sgd_pre_split_hdf5", save_to="models/adience_xent_l2-1e-4_sgd_pre_split_hdf5", debug=False)

    # adience_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.40.bak
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="x_ent",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)),"momentum":0.9}, debug=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_xent_l2-1e-4_sgd_pre_split_hdf5", save_to="models/adience_xent_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/adience_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.40.bak")

def adience_pois_t_1_xent_l2_1e4_sgd_pre_split_hdf5(mode):
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience_pois, num_classes=8, mode="x_ent",
                   args={"l2":1e-4, "tau":1.0, "end_nonlinearity":softplus}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)),"momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/adience_256.h5", "/data/lisatmp4/beckhamc/hdf5/adience_256.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    name = "adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5"
    if mode == "train":
        nn.train(
            data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
            iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
            out_dir="output/%s" % name, save_to="models/%s" % name, debug=False)

def adience_pois_t_5_xent_l2_1e4_sgd_pre_split_hdf5(mode):
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience_pois, num_classes=8, mode="x_ent",
                   args={"l2":1e-4, "tau":0.5, "end_nonlinearity":softplus}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)),"momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/adience_256.h5", "/data/lisatmp4/beckhamc/hdf5/adience_256.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    name = "adience_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5"
    if mode == "train":
        nn.train(
            data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
            iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
            out_dir="output/%s" % name, save_to="models/%s" % name, debug=False)
        
def adience_exp_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="exp",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.1)),"momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/adience_256.h5", "/data/lisatmp4/beckhamc/hdf5/adience_256.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/adience_exp_l2-1e-4_sgd_pre_split_hdf5", save_to="models/adience_exp_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/adience_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.100.bak2")
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="exp",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)),"momentum":0.9}, debug=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_exp_l2-1e-4_sgd_pre_split_hdf5", save_to="models/adience_exp_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/adience_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.16.bak3")
    
    # adience_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.16.bak3

def adience_emd2_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="emd2",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)),"momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/adience_256.h5", "/data/lisatmp4/beckhamc/hdf5/adience_256.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/adience_emd2_l2-1e-4_sgd_pre_split_hdf5", save_to="models/adience_emd2_l2-1e-4_sgd_pre_split_hdf5", debug=False)
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="emd2",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)),"momentum":0.9}, debug=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_emd2_l2-1e-4_sgd_pre_split_hdf5", save_to="models/adience_emd2_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/adience_emd2_l2-1e-4_sgd_pre_split_hdf5.modelv1.76.bak")
    
def adience_emd2_l2_1e4_adam_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="emd2",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_emd2_l2-1e-4_adam_pre_split", save_to="models/adience_emd2_l2-1e-4_adam_pre_split", debug=False)

def adience_emd22_l2_1e4_adam_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="emd22",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_emd22_l2-1e-4_adam_pre_split", save_to="models/adience_emd22_l2-1e-4_adam_pre_split", debug=False)

def adience_qwk_l2_1e4_adam_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, mode="qwk",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_qwk_l2-1e-4_adam_pre_split", save_to="models/adience_qwk_l2-1e-4_adam_pre_split", debug=False)

##################
# DR EXPERIMENTS #
##################

def dr_xent_l2_1e4_adam_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    nn.train(
        data=datasets.dr.load_pre_split_data("/data/lisatmp4/beckhamc/train-trim-ben-256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=250,
        out_dir="output/dr_xent_l2-1e-4_adam_pre_split", save_to="models/dr_xent_l2-1e-4_adam_pre_split", debug=False, resume="models/dr_xent_l2-1e-4_adam_pre_split.modelv1.10.bak")

def dr_xent_l2_1e4_adam_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5("/data/lisatmp4/beckhamc/hdf5/dr.h5"),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=250,
        out_dir="output/dr_xent_l2-1e-4_adam_pre_split_hdf5", save_to="models/dr_xent_l2-1e-4_adam_pre_split_hdf5", debug=False)

def dr_soft_l2_1e4_adam_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="soft",
                   args={"l2":1e-4, "y_soft_sigma":1.}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5("/data/lisatmp4/beckhamc/hdf5/dr.h5"),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=250,
        out_dir="output/dr_soft_l2-1e-4_adam_pre_split_hdf5", save_to="models/dr_soft_l2-1e-4_adam_pre_split_hdf5", debug=False)

def dr_soft5_l2_1e4_adam_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="soft",
                   args={"l2":1e-4, "y_soft_sigma":0.5}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5("/data/lisatmp4/beckhamc/hdf5/dr.h5"),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=250,
        out_dir="output/dr_soft0.5_l2-1e-4_adam_pre_split_hdf5", save_to="models/dr_soft0.5_l2-1e-4_adam_pre_split_hdf5", debug=False)

    
    
def dr_xent_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr.h5", "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
    #    out_dir="output/dr_xent_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_xent_l2-1e-4_sgd_pre_split_hdf5", debug=False)
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)), "momentum":0.9}, debug=True)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
    #    out_dir="output/dr_xent_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_xent_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/dr_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.100.bak")

    model_name = "dr_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.44.bak2"
    nn.load_weights_from("models/%s" % model_name)
    xt, yt, xv, yv = datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file)
    nn.dump_dists(xv, yv, iterators.iterate_hdf5(imgen,224), 128, "dists/%s.csv" % model_name)
 
def dr_exp_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.1)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr.h5", "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5", debug=False)

    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.1)), "momentum":0.9}, debug=True)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.33.bak")
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak2")
    #dr_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak2
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)), "momentum":0.9}, debug=True)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.55.bak3")
    
    model_name = "dr_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak4"
    nn.load_weights_from("models/%s" % model_name)
    xt, yt, xv, yv = datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file)
    nn.dump_dists(xv, yv, iterators.iterate_hdf5(imgen,224), 128, "dists/%s.csv" % model_name)

    # dr_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak4
 
def dr_emd2_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="emd2",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr.h5", "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/dr_emd2_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_emd2_l2-1e-4_sgd_pre_split_hdf5", debug=False)
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="emd2",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)), "momentum":0.9}, debug=True)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=100,
    #    out_dir="output/dr_emd2_l2-1e-4_sgd_pre_split_hdf5", save_to="models/dr_emd2_l2-1e-4_sgd_pre_split_hdf5", debug=False, resume="models/dr_emd2_l2-1e-4_sgd_pre_split_hdf5.modelv1.100.bak")

    model_name = "dr_emd2_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak2"
    nn.load_weights_from("models/%s" % model_name)
    xt, yt, xv, yv = datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file)
    nn.dump_dists(xv, yv, iterators.iterate_hdf5(imgen,224), 128, "dists/%s.csv" % model_name)

    
    
def dr_exp_l2_1e4_adam_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5("/data/lisatmp4/beckhamc/hdf5/dr.h5"),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=250,
        out_dir="output/dr_exp_l2-1e-4_adam_pre_split_hdf5", save_to="models/dr_exp_l2-1e-4_adam_pre_split_hdf5", debug=False, resume="models/dr_exp_l2-1e-4_adam_pre_split_hdf5.modelv1.28.bak2")

def dr_emd2_l2_1e4_adam_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="emd2",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr.h5", "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=250,
        out_dir="output/dr_emd2_l2-1e-4_adam_pre_split_hdf5", save_to="models/dr_emd2_l2-1e-4_adam_pre_split_hdf5", debug=False)

def dr_xent_l2_1e4_adam_pre_split_hdf5_fuel():
    lasagne.random.set_rng(np.random.RandomState(1))
    # default learning rate for adam is 0.001
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
                   args={"l2":1e-4}, optimiser=adam, optimiser_args={"learning_rate":theano.shared(floatX(0.001))}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dataset_dir = "/data/lisatmp4/beckhamc/hdf5/dr_fuel_test.h5"
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5_fuel(dataset_dir),
        iterator_fn=iterators.iterate_hdf5_fuel(dataset_dir, imgen, 224), batch_size=128, num_epochs=250,
        out_dir="output/dr_xent_l2-1e-4_adam_pre_split_hdf5_fuel", save_to="models/dr_xent_l2-1e-4_adam_pre_split_hdf5_fuel", debug=False)

# dr with 0.95 split on validation set

def dr_xent_l2_1e4_sgd_pre_split_hdf5_v95():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr_s0.95.h5", "/data/lisatmp4/beckhamc/hdf5/dr_s0.95.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
    #    out_dir="output/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95", save_to="models/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95", debug=False)
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)), "momentum":0.9}, debug=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
        out_dir="output/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95", save_to="models/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95", debug=False, resume="models/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95.modelv1.100.bak")

def dr_xent_l2_1e4_sgd_pre_split_hdf5_v95_async():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="x_ent",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr_s0.95.h5", "/data/lisatmp4/beckhamc/hdf5/dr_s0.95.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_semi_shuffle_async(imgen,224), batch_size=128, num_epochs=200,
        out_dir="output/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95_async", save_to="models/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95_async", debug=False)
    
def dr_exp_l2_1e4_sgd_pre_split_hdf5_v95():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.1)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr_s0.95.h5", "/data/lisatmp4/beckhamc/hdf5/dr_s0.95.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
    #    out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95", debug=False)
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
    #    out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95", debug=False, resume="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95.modelv1.53.bak")
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)), "momentum":0.9}, debug=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
        out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95", debug=False, resume="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95.modelv1.47.bak2")

def dr_exp_l2_1e4_sgd_pre_split_hdf5_v95_repeat():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.1)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr_s0.95.h5", "/data/lisatmp4/beckhamc/hdf5/dr_s0.95.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
    #    out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95_repeat", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95_repeat", debug=False)
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="exp",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
        out_dir="output/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95_repeat", save_to="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95_repeat", debug=False,
        resume="models/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95_repeat.modelv1.41.bak")
    
def dr_emd2_l2_1e4_sgd_pre_split_hdf5_v95():
    lasagne.random.set_rng(np.random.RandomState(1))
    #nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="emd2",
    #               args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr_s0.95.h5", "/data/lisatmp4/beckhamc/hdf5/dr_s0.95.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    #nn.train(
    #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
    #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
    #    out_dir="output/dr_emd2_l2-1e-4_sgd_pre_split_hdf5_v95", save_to="models/dr_emd2_l2-1e-4_sgd_pre_split_hdf5_v95", debug=False)
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr, num_classes=5, mode="emd2",
                   args={"l2":1e-4}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.001)), "momentum":0.9}, debug=True)
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
        out_dir="output/dr_emd2_l2-1e-4_sgd_pre_split_hdf5_v95", save_to="models/dr_emd2_l2-1e-4_sgd_pre_split_hdf5_v95", debug=False, resume="models/dr_emd2_l2-1e-4_sgd_pre_split_hdf5_v95.modelv1.26.bak2")

    
# ####################

def dr_pois_t_1_xent_l2_1e4_sgd_pre_split_hdf5(mode):
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr_pois, num_classes=5, mode="x_ent",
            args={"l2":1e-4, "tau":1.0, "end_nonlinearity":softplus}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr.h5", "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    name = "dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5"
    assert mode in ["train", "dump_dist", "dump_fx"]
    if mode == "train":
        #nn.train(
        #    data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        #    iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
        #    out_dir="output/%s" % name, save_to="models/%s" % name, debug=False)
        pass
    elif mode == "dump_dist":
        model_name = "dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.200.bak"
        nn.load_weights_from("models/%s" % model_name)
        xt, yt, xv, yv = datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file)
        nn.dump_dists(xv, yv, iterators.iterate_hdf5(imgen,224), 128, "dists/%s.csv" % model_name)
    elif mode == "dump_fx":
        model_name = "dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.200.bak"
        nn.load_weights_from("models/%s" % model_name)
        xt, yt, xv, yv = datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file)
        nn.dump_output_for_layer(nn.l_out.input_layer.input_layer.input_layer, xt, yt, iterators.iterate_hdf5(imgen,224), 128, "dists/%s.fx.csv" % model_name)

def dr_pois_t_5_xent_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr_pois, num_classes=5, mode="x_ent",
            args={"l2":1e-4, "tau":0.5, "end_nonlinearity":softplus}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr.h5", "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    name = "dr_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5"
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
        out_dir="output/%s" % name, save_to="models/%s" % name, debug=False)

def dr_pois_t_3_xent_l2_1e4_sgd_pre_split_hdf5():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_dr_pois, num_classes=5, mode="x_ent",
            args={"l2":1e-4, "tau":0.3, "end_nonlinearity":softplus}, optimiser=nesterov_momentum, optimiser_args={"learning_rate":theano.shared(floatX(0.01)), "momentum":0.9}, debug=True)
    imgen = ImageDataGenerator(rotation_range=359.,width_shift_range=0.05,height_shift_range=0.05,zoom_range=0.02,fill_mode='constant',cval=0.5,horizontal_flip=True,vertical_flip=True)
    dest_file, from_file = "/Tmp/beckhamc/hdf5/dr.h5", "/data/lisatmp4/beckhamc/hdf5/dr.h5"
    copy_if_not_exist(dest_file=dest_file, from_file=from_file)
    name = "dr_pois_t-0.3_xent_l2-1e-4_sgd_pre_split_hdf5"
    nn.train(
        data=datasets.dr.load_pre_split_data_into_memory_as_hdf5(dest_file),
        iterator_fn=iterators.iterate_hdf5(imgen, 224), batch_size=128, num_epochs=200,
        out_dir="output/%s" % name, save_to="models/%s" % name, debug=False)








##########################
    
def adience_xemd2_1e1_pre_split():
    lasagne.random.set_rng(np.random.RandomState(1))
    nn = NeuralNet(architectures.resnet.resnet_2x4_adience, num_classes=8, learning_rate=0.01, momentum=0.9, mode="xemd2",
                   args={"emd2_lambda":1e-1}, debug=True)
    imgen = ImageDataGenerator(horizontal_flip=True)
    nn.train(
        data=datasets.adience.load_pre_split_data("/data/lisatmp4/beckhamc/adience_data/aligned_256x256"),
        iterator_fn=iterators.iterate_filenames(imgen, 224), batch_size=128, num_epochs=100,
        out_dir="output/adience_xemd2_1e-1_pre_split", save_to="models/adience_xemd2_1e-1_pre_split", debug=False)

    
    
if __name__ == '__main__':

    np.random.seed(0)

    locals()[ sys.argv[1] ]( sys.argv[2] )

    #mnist_earth_mover()
    #mnist_exp()
    #mnist_baseline()

    #adience_baseline_f0()
    #adience_baseline_pre_split()
