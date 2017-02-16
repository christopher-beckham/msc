# -------------
# plot dists
# -------------

d1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.200.bak.csv",header=F)
#d1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.44.bak2.csv",header=F)
set.seed(100); idxs = sample(1:nrow(d1), 4*4)
par(mfrow=c(4,4))
labels=c()
for(i in idxs) {
  barplot(as.numeric(d1[i,1:5]), names.arg=0:4) 
  labels = c(labels, d1[i,6])
}
print(labels)
