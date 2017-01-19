import iterators
import datasets.adience



def train(data, iterator_fn):
    xt, yt, xv, y = data
    for xb, yb in iterator_fn(xt, yt, bs=32, num_classes=8):
        print xb.shape, yb.shape
        break


if __name__ == '__main__':
    train(datasets.adience.get_fold(0), iterators.iterate_filenames)