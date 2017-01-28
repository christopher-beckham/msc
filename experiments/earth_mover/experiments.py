import sys
import os
import architectures
import architectures.simple
import architectures.resnet
import datasets
import datasets.mnist
import datasets.adience
import datasets.dr
import iterators
from base import NeuralNet
import numpy as np
import lasagne
from lasagne.updates import *
from lasagne.utils import *
import theano
from keras.preprocessing.image import ImageDataGenerator

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

    
    
# ####################
    
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

    locals()[ sys.argv[1] ]()

    #mnist_earth_mover()
    #mnist_exp()
    #mnist_baseline()

    #adience_baseline_f0()
    #adience_baseline_pre_split()
