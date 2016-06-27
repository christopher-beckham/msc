import cv2, glob, numpy
import sys
import os

prefix = os.environ["DATA_DIR"]
src_dir = sys.argv[1]
dest_dir = sys.argv[2]

def scaleRadius(img, scale):
    x = img[img.shape[0]/2,:,:].sum(1)
    r = (x>x.mean()/10).sum()/2

    # hacky fix for corrupted images??
    if r == 0:
        return img

    s=scale*1.0/r
    return cv2.resize(img, (0,0), fx=s, fy=s)


scale = 400
for f in glob.glob( prefix + os.path.sep + src_dir + os.path.sep + "*.jpeg"):
    #print f
    a = cv2.imread(f)
    a = scaleRadius(a,scale)
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
    b = numpy.zeros(a.shape)
    cv2.circle(b, (a.shape[1]/2, a.shape[0]/2), int(scale*0.9), (1,1,1),-1,8,0)
    a = a*b+128*(1-b)
    
    cv2.imwrite(prefix + os.path.sep + dest_dir + os.path.sep + os.path.basename(f),a)
    print prefix + os.path.sep + dest_dir + os.path.sep + os.path.basename(f)
