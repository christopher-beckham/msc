import cPickle as pickle
import sys
import os
import glob

X_left = []
X_right = []
tmp = {}
for filename in glob.glob("/data/lisatmp4/beckhamc/test-trim-ben-r400-512/*.jpeg"):
    #if "38343" not in filename:
    #    continue
    basename = os.path.basename(filename).replace(".jpeg","")
    #print "basename", basename
    keyname = basename.replace("_left","").replace("_right","")
    #print "keyname", keyname
    if keyname not in tmp:
        tmp[keyname] = {"left":None, "right":None}

    if "left" in basename:
        tmp[keyname]["left"] = basename
    elif "right" in basename:
        tmp[keyname]["right"] = basename
    else:
        raise Exception("left/right error")

X_left=[]
X_right=[]
for key in tmp:
    assert tmp[key]["left"] != None or tmp[key]["right"] != None
    X_left.append(tmp[key]["left"])
    X_right.append(tmp[key]["right"])

for xl, xr in zip(X_left, X_right):
    print xl, xr
