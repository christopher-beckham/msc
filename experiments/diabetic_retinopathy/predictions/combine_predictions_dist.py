import sys
import os
from collections import OrderedDict, Counter
import numpy as np


dd = OrderedDict()
filename = sys.argv[1]
with open(filename) as f:
    f.readline() # skip header
    for line in f:
        line = line.rstrip().split(",")
        name = line[0]
        dist = [ float(st) for st in line[1::] ]
        if name not in dd:
            dd[name] = []
        dd[name].append(dist)

#for key in dd:
#    assert len(dd[key]) == 40

print "image,level"
for key in dd:
    mat = np.asarray(dd[key])
    #print key
    #print mat

    avg_dist = [ np.mean(mat[:,0]), np.mean(mat[:,1]), np.mean(mat[:,2]), np.mean(mat[:,3]), np.mean(mat[:,4]) ]
    #print avg_dist
    pred = int(round(np.dot(avg_dist, np.arange(0,5))))
    print key + "," + str(pred)
