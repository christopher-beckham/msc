adx.dist1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/adience_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.60.bak2.valid.csv",header=F)

adxp.dist1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.100.bak.valid.csv",header=F)


# compute accuracy for x-ent
tmp = c()
for(i in 1:nrow(adx.dist1)) {
  tmp = c(tmp, which.max(as.numeric(adx.dist1[i,1:8]))-1)
}
print(mean(adx.dist1[,9] == tmp))

# compute accuracy for x-ent + pois
