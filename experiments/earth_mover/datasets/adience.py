import sys
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
# --
from adience_test import get_fold, load_image_keras

if __name__ == '__main__':
    # testing
    xt, yt, xv, yv = get_fold(1)
    print xt[0]
    print load_image_keras(xt[0], crop=224)