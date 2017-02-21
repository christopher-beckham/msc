import h5py
import numpy as np
import matplotlib.pyplot as plt
import dr
import pdb
from scipy.misc import imsave

if __name__ == '__main__':
    h5_src = "/data/lisatmp4/beckhamc/hdf5/adience_256.h5"
    out_dir = "adience_valid_dump"
    xt, yt, xv, yv = dr.load_pre_split_data_into_memory_as_hdf5(h5_src)
    #print xt.shape, yt.shape, xv.shape, yv.shape

    for i in range(xv.shape[0]):
        cls = yv[i]
        print "saving image %i" % i
        new_img = xv[i].swapaxes(0,1).swapaxes(1,2)
        imsave("%s/%i_%i.png" % (out_dir, i, cls), new_img)
