# -------------
# plot dists
# -------------

#d1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.200.bak.csv",header=F)
#d1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_emd2_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak2.csv",header=F)

prep.margins = function() {
  par(mar=c(2,2,2.5,1)+0.1) 
}

# CROSS-ENTROPY DISTRIBUTIONS ON THE VALID SET
pdf("dr_xent_dists.pdf",height=4,width=6)
d1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.44.bak2.csv",header=F)
set.seed(100); idxs = sample(1:nrow(d1), 4*4)
par(mfrow=c(4,4))
prep.margins()
labels=c()
for(i in idxs) {
  barplot(as.numeric(d1[i,1:5]), names.arg=0:4) 
  labels = c(labels, d1[i,6])
}
print(labels)
dev.off()

# POIS (TAU=1) DISTRIBUTIONS ON THE VALID SET
pdf("dr_pois_dists.pdf",height=4,width=6)
d2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.200.bak.csv",header=F)
set.seed(100); idxs = sample(1:nrow(d1), 4*4)
par(mfrow=c(4,4))
prep.margins()
labels=c()
for(i in idxs) {
  barplot(as.numeric(d2[i,1:5]), names.arg=0:4) 
  labels = c(labels, d2[i,6])
}
print(labels)
dev.off()





