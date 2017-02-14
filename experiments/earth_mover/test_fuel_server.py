import fuel
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import MultiProcessing
from fuel.streams import DataStream
from fuel.server import start_server

import theano
import numpy as np
import sys

def dr_train_iterator_sequential():
    #dataset_dir = "/data/lisatmp4/beckhamc/hdf5/dr_fuel_test.h5"
    #X_id = "train"
    #bs = 128

    dataset_dir = "/data/lisatmp4/beckhamc/hdf5/dummy.h5"
    X_id = "train"
    bs = 3
    dataset = H5PYDataset(dataset_dir, which_sets=(X_id, ))
    iterator_scheme = SequentialScheme(batch_size=bs, examples=dataset.num_examples)
    #ds = MultiProcessing(DataStream(dataset=dataset, iteration_scheme=iterator_scheme), max_store=20)
    start_server(DataStream(dataset=dataset, iteration_scheme=iterator_scheme))

def test():
    dataset_dir = "/data/lisatmp4/beckhamc/hdf5/dummy.h5"
    X_id = "train"
    bs = 3
    dataset = H5PYDataset(dataset_dir, which_sets=(X_id, ))
    iterator_scheme = SequentialScheme(batch_size=bs, examples=dataset.num_examples)
    ds = DataStream(dataset=dataset, iteration_scheme=iterator_scheme)
    for data in ds.get_epoch_iterator():
        print np.argmax(data[0],axis=1)
    for data in ds.get_epoch_iterator():
        print np.argmax(data[0],axis=1)
        
    
    
if __name__ == '__main__':
    locals()[ sys.argv[1] ]()
