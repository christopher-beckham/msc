import fuel
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import MultiProcessing
from fuel.streams import DataStream, ServerDataStream
from fuel.server import start_server

import theano
import numpy as np
from collections import Counter

total_y = []
ds = ServerDataStream((),produces_examples=False)
#ds.reset() # not implemented

"""
for data in ds.get_epoch_iterator():
    xb, yb = data
    total_y += yb.tolist()
    print len(total_y)

cc = Counter(total_y)
print cc.most_common()
"""

for data in ds.get_epoch_iterator():
    print np.argmax(data[0],axis=1)


ds2 = ServerDataStream((),produces_examples=False)

for data in ds2.get_epoch_iterator():
    print np.argmax(data[0],axis=1)
