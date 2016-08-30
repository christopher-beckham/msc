import sys
import os
from collections import OrderedDict, Counter

dd = OrderedDict()
filename = sys.argv[1]
with open(filename) as f:
    f.readline() # skip header
    for line in f:
        line = line.rstrip().split(",")
        name = line[0]
        cls = line[1]
        if name not in dd:
            dd[name] = []
        dd[name].append(cls)

#for key in dd:
#    assert len(dd[key]) == 40

print "image,level"
for key in dd:
    most_common = Counter(dd[key]).most_common()[0][0]
    print key + "," + most_common
