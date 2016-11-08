import quadrant_network_dr
import numpy as np
import theano
from theano import tensor as T

def random_prob_dist(n, k=5):
    mat = np.abs(np.random.normal(0,1,size=(n,k)))
    for i in range(n):
        mat[i,] = mat[i,] / np.sum(mat[i,])
    return mat.astype("float32")

def random_one_hot(n):
    tot = []
    for i in range(n):
        arr = [0,0,0,0,0]
        arr[ np.random.randint(0,5) ] = 1
        tot.append(arr)
    return np.asarray(tot, dtype="float32")

print quadrant_network_dr.hp.weighted_kappa
print quadrant_network_dr.qwk

y1 = T.fmatrix('y1')
y2 = T.fmatrix('y2')


# seems to be equivalent 98% of the time -- the cases
# where it fails are tests between numbers like Xe-16 and Xe-08,
# and these are so close to 0 anyway...
for x in range(0, 100):
    yvector1 = random_one_hot(10)
    yvector2 = random_one_hot(10)
    cvector1 = np.argmax(yvector1,axis=1)
    cvector2 = np.argmax(yvector2,axis=1)
    a = quadrant_network_dr.hp.weighted_kappa(cvector1, cvector2)
    b = 1-quadrant_network_dr.qwk(y1,y2).eval({y1:yvector1, y2:yvector2})
    print a, b, np.isclose(a,b)
    #print yvector1
    #print yvector2
    #print quadrant_network_dr.qwk(y1,y2).eval({y1:yvector1, y2:yvector2})

