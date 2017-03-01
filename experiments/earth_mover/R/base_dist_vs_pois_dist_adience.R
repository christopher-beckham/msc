adx.dist1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/adience_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.60.bak2.valid.csv",header=F)

adxp.dist1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.100.bak.valid.csv",header=F)

g.like = function(arr, which.idx) {
  b1 = all( sort(arr[which.idx:length(arr)],decreasing=T) == arr[which.idx:length(arr)] )
  b2 = all( arr[1:which.idx] == sort(arr[1:which.idx]) )
  return(all(c(b1,b2)))
}

for(i in 1:nrow(adx.dist1)) {
  dist = adx.dist1[i,1:8]
  cls = adx.dist1[i,9]
  print(g.like(dist,which.max(dist)))
}


# handpicked idxs
idxs = c(426,223,112,100,31,17,19,500,609,611,713,714,711,701,901,911)

# CROSS-ENTROPY DISTRIBUTIONS ON THE VALID SET
pdf("adience_xent_dists.pdf",height=4,width=5)
#idxs = 420:435 # 16-31
num.classes = 8
par(mfrow=c(4,4))
prep.margins()
labels=c()
for(i in idxs) {
  barplot(as.numeric(adx.dist1[i,1:num.classes]), names.arg=0:(num.classes-1),main=i-1) 
  labels = c(labels, adx.dist1[i,num.classes+1])
}
print(labels)
dev.off()

# CROSS-ENTROPY + POIS DISTRIBUTIONS ON THE VALID SET
pdf("adience_xent-pois_dists.pdf",height=4,width=5)
#idxs = 420:435 # 16-31
num.classes = 8
par(mfrow=c(4,4))
prep.margins()
labels=c()
for(i in idxs) {
  barplot(as.numeric(adxp.dist1[i,1:num.classes]), names.arg=0:(num.classes-1),main=i-1) 
  labels = c(labels, adxp.dist1[i,num.classes+1])
}
print(labels)
dev.off()


# -----------

# compute accuracy for x-ent
preds.xent = c()
for(i in 1:nrow(adx.dist1)) {
  preds.xent = c(preds.xent, which.max(as.numeric(adx.dist1[i,1:8]))-1)
}
print(mean(adx.dist1[,9] == preds.xent))

# compute accuracy for x-ent + pois
preds.xpois = c()
for(i in 1:nrow(adxp.dist1)) {
  preds.xpois = c(preds.xpois, which.max(as.numeric(adxp.dist1[i,1:8]))-1)
}
print(mean(adxp.dist1[,9] == preds.xpois))

which.wrong.xent = which( (preds.xent == adxp.dist1[,9] ) == FALSE) # which idxs were misclassified by xent
which.wrong.xpois = which( (preds.xpois == adxp.dist1[,9] ) == FALSE) # which idxs were misclassified by xpois

# which idxs were misclassified by xpois and *not* misclassified by xent
set.diff = setdiff(which.wrong.xpois, which.wrong.xent)


