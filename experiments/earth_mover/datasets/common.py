import numpy as np

def group_indices_by_class(y):
    dd = {}
    for i in range(y.shape[0]):
        if y[i] not in dd:
            dd[ y[i] ] = []
        dd[ y[i] ].append(i)
    return dd

def create_imbalanced_data(X_train, y_train, idxs_by_class, classes_to_reduce, reduce_by, seed=0):
    """
    :idxs_by_class: a dict mapping class indices to indices
    :classes_to_reduce: which classes should we randomly remove indices from?
    :reduce_by: by what factor should we reduce? e.g. 0.5 = remove 50% of the indices
    """
    rnd_state = np.random.RandomState(seed)
    tot_idxs = []
    for key in idxs_by_class:
        idxs_shuffled = idxs_by_class[key][:] # [:] to copy
        rnd_state.shuffle(idxs_shuffled)
        if key in classes_to_reduce:
            idxs_shuffled = idxs_shuffled[0:int(len(idxs_shuffled)*reduce_by)]
        tot_idxs += idxs_shuffled
    return X_train[tot_idxs], y_train[tot_idxs]

