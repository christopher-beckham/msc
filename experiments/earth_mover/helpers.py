import numpy as np
from theano import tensor as T
from scipy.stats import norm

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

def one_hot_to_soft(y, sigma=1.0):
    # TODO: vectorise
    y_softs = np.zeros(y.shape, dtype=y.dtype)
    for i in range(y.shape[0]):
        this_y = np.zeros(y.shape[1], dtype=y.dtype)
        for j in range(y.shape[1]):
            this_y[j] = norm.pdf(j, np.argmax(y[i]), sigma)
        y_softs[i] = this_y
        y_softs[i] = y_softs[i] / np.sum(y_softs[i])
    return y_softs

def label_to_one_hot(this_y, num_classes):
    this_onehot = []
    for elem in this_y:
        one_hot_vector = [0]*num_classes
        one_hot_vector[elem] = 1
        this_onehot.append(one_hot_vector)
    this_onehot = np.asarray(this_onehot, dtype="float32")
    return this_onehot

def one_hot_to_label(x):
    assert len(x.shape) == 2
    return np.argmax(x,axis=1).astype("int32")

def qwk(predictions, targets, num_classes=5):
    w = np.ones((num_classes,num_classes)).astype("float32")
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            # NOTE: this differs from diabetic_retinopathy qwk
            # in the sense that w_ij is normalised
            w[i,j] = ((i-j)**2) / (num_classes-1)
    numerator = T.dot(targets.T, predictions)
    denominator = T.outer(targets.sum(axis=0), predictions.sum(axis=0))
    denominator = denominator / T.sum(numerator)
    qwk = (T.sum(numerator*w) / T.sum(denominator*w))
    return qwk

def weighted_kappa(human_rater, actual_rater, num_classes=5):
    assert len(human_rater) == len(actual_rater)
    def sum_matrix(X, Y):
        assert len(X) == len(Y)
        assert len(X[0]) == len(Y[0])
        sum = 0
        for i in range(0, len(X)):
            for j in range(0, len(X[0])):
                sum += X[i][j]*Y[i][j]
        return sum
    # compute W
    W = [ [float(0) for y in range(0, num_classes)] for x in range(0, num_classes) ]
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            num = (i-j)**2
            den = (float(num_classes) - 1)**2
            W[i][j] = num # / den
    # compute O
    O = [ [float(0) for y in range(0, num_classes)] for x in range(0, num_classes) ]
    # rows = human_rater
    # cols = actual_rater
    for i in range(0, len(actual_rater)):
        O[ human_rater[i] ][ actual_rater[i] ] += 1
    # normalise O
    total = sum([sum(x) for x in O])
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            O[i][j] = O[i][j] / total
    # compute E
    total = sum([sum(x) for x in O])
    E = [ [float(0) for y in range(0, num_classes)] for x in range(0, num_classes) ]
    for i in range(0, num_classes):
        for j in range(0, num_classes):
            # E_ij = row(i) total * col(j) total / total
            col_j = [ O[x][j] for x in range(0, len(O[0])) ]
            E[i][j] = sum(O[i]) * sum(col_j) / total
    # compute kappa
    kappa = 1 - (sum_matrix(W, O) / sum_matrix(W, E))
    return kappa

if __name__ == '__main__':
    x = np.asarray([[1,0,0],[0,1,0],[0,0,1]]).astype("float32")
    #one_hot_to_cmf(x)

    print one_hot_to_ord(x)
