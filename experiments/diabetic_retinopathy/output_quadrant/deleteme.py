import sys

with open(sys.argv[1]) as f:
  for line in f:
    arr = line.rstrip().split(",")
    print ",".join(arr[0:len(arr)-1]) + ",," + str(arr[-1])
