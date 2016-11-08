import numpy as np
import cPickle as pickle
from itertools import cycle

with open("dr.pkl") as f:
    dat = pickle.load(f)
X_left, X_right, y_left, y_right = dat
X_left = np.asarray(X_left)
X_right = np.asarray(X_right)
y_left = np.asarray(y_left, dtype="int32")
y_right = np.asarray(y_right, dtype="int32")
np.random.seed(0)
idxs = [x for x in range(0, len(X_left))]
np.random.shuffle(idxs)

X_train_left = X_left[idxs][0 : int(0.95*X_left.shape[0])]
X_train_right = X_right[idxs][0 : int(0.95*X_right.shape[0])]
y_train_left = y_left[idxs][0 : int(0.95*y_left.shape[0])]
y_train_right = y_right[idxs][0 : int(0.95*y_left.shape[0])]
X_valid_left = X_left[idxs][int(0.95*X_left.shape[0]) ::]
X_valid_right = X_right[idxs][int(0.95*X_right.shape[0]) ::]
y_valid_left = y_left[idxs][int(0.95*y_left.shape[0]) ::]
y_valid_right = y_right[idxs][int(0.95*y_right.shape[0]) ::]

# ok, fix now
X_train = np.hstack((X_train_left, X_train_right))
X_valid = np.hstack((X_valid_left,X_valid_right))
y_train = np.hstack((y_train_left,y_train_right))
y_valid = np.hstack((y_valid_left,y_valid_right))

num_classes=5

dd = dict()
for i in range(0,num_classes):
    dd[i] = []
for filename, class_idx in zip(X_train, y_train):
    dd[class_idx].append(filename)

iterators = []
for i in range(0,num_classes):
    iterators.append( cycle(dd[i]) )

# ok, let's prepare a batch

for num_iters in range(100):

    this_X = []
    this_y = []
    bs = 128
    for i in range(0, len(iterators)):
        this_X += [iterators[i].next() for j in range(bs//num_classes)]
        this_y += [i]*(bs//num_classes)

    print this_X
    print this_y
