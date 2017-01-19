import numpy as np

def one_hot_to_cmf(x):
    """
    convert a one-hot distribution to a cmf
    :param x:
    :return:
    """
    assert len(x.shape) == 2
    W = np.ones((x.shape[1], x.shape[1]), dtype="float32")
    for k in range(0, x.shape[1]):
        W[k][0:k] = 0
    return np.dot(x, W)

def one_hot_to_ord(x):
    """
    convert a one-hot distribution to an ord
    :param x:
    :return:
    """
    assert len(x.shape) == 2
    num_classes = x.shape[1]
    tot = []
    for row in x:
        k = np.argmax(row)
        arr = [0]*(num_classes-1)
        if k == 0:
            tot.append(arr)
            continue
        for i in range(0, k):
            arr[i] = 1
        tot.append(arr)
    tot = np.asarray(tot, dtype=x.dtype)
    return tot

if __name__ == '__main__':
    x = np.asarray([[1,0,0],[0,1,0],[0,0,1]]).astype("float32")
    #one_hot_to_cmf(x)

    print one_hot_to_ord(x)
