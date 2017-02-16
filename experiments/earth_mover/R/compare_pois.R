df =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_sgd_pre_split_hdf5_v95/results.txt")
df2 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95/results.txt")
df3 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_emd2_l2-1e-4_sgd_pre_split_hdf5_v95/results.txt")

df4 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")
df5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")

plot(df$train_loss,type="l")
lines(df2$train_loss,col="red")
lines(df3$train_loss,col="blue")

plot(df$valid_xent,type="l", xlim=c(0,200))
lines(df2$valid_xent,col="red")
lines(df3$valid_xent,col="blue")

plot(df$valid_exp_qwk,type="l", xlim=c(0,200))
lines(df2$valid_exp_qwk,col="red")
lines(df3$valid_exp_qwk,col="blue")

plot(df$valid_xent_qwk,type="l", xlim=c(0,200))
lines(df2$valid_xent_qwk,col="red")
lines(df3$valid_xent_qwk,col="blue")

# --------

df2b =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_exp_l2-1e-4_sgd_pre_split_hdf5_v95_repeat//results.txt")

# --------

df =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
df4 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_dlra100//results.txt")
df5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
df6 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.3_xent_l2-1e-4_sgd_pre_split_hdf5_dlra100/results.txt")

par(mfrow=c(1,1))

plot(df$valid_xent,type="l",xlim=c(0,200))
lines(df4$valid_xent,col="red")
#lines(df5$valid_xent,col="blue")
lines(df6$valid_xent,col="purple")

plot(df$valid_xent_qwk,type="l",xlim=c(0,200))
lines(df4$valid_xent_qwk,col="red")
#lines(df5$valid_xent_qwk,col="blue")
lines(df6$valid_xent_qwk,col="purple")

plot(df$valid_exp_qwk,type="l")
lines(df4$valid_exp_qwk,col="red")
#lines(df5$valid_exp_qwk,col="blue")
lines(df6$valid_exp_qwk,col="purple")

plot(df4$valid_loss,col="red",type="l")

plot(df5$valid_loss,col="blue",type="l")

plot(df6$valid_loss,col="purple",type="l")


tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")
tmp2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")
tmp3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.3_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")
tmpe = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

par(mfrow=c(1,1))

plot(tmp$valid_loss,type="l")
lines(tmp2$valid_loss,col="red")
lines(tmp3$valid_loss,col="blue")

plot(tmp$valid_xent,type="l",ylim=c(1,2))
lines(tmp2$valid_xent,col="red")
lines(tmp3$valid_xent,col="blue")
lines(tmpe$valid_xent,col="purple")

plot(tmp$valid_xent_qwk,type="l")
#lines(tmp2$valid_xent_qwk,col="red")
#lines(tmp3$valid_xent_qwk,col="blue")
lines(tmpe$valid_xent_qwk,col="purple")

plot(tmp$valid_exp_qwk,type="l")
#lines(tmp2$valid_exp_qwk,col="red")
#lines(tmp3$valid_exp_qwk,col="blue")
lines(tmpe$valid_exp_qwk,col="purple")

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

# -------------
# look at f(x)
# -------------

library(reshape2)
library(ggplot2)


f1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.200.bak.fx.csv",header=F)

#p0 = hist(f1[ f1$V2 == 0,]$V1,breaks=20)
#p1 = hist(f1[ f1$V2 == 1,]$V1,breaks=20)
#p2 = hist(f1[ f1$V2 == 2,]$V1,breaks=20)
#p3 = hist(f1[ f1$V2 == 3,]$V1,breaks=20)
#p4 = hist(f1[ f1$V2 == 4,]$V1,breaks=20)
#plot(p0, col=rgb(1,0,0,1/4), xlim=c(0,40))
#plot(p1, col=rgb(0,1,0,1/4), xlim=c(0,40), add=T)

par(mfrow=c(1,1))

plot(density(f1[ f1$V2 == 0,]$V1),ylim=c(0,1))
lines(density(f1[ f1$V2 == 1,]$V1),col="red")
lines(density(f1[ f1$V2 == 2,]$V1),col="orange")
lines(density(f1[ f1$V2 == 3,]$V1),col="blue")
lines(density(f1[ f1$V2 == 4,]$V1),col="purple")

boxplot(
  f1[ f1$V2 == 0,]$V1,
  f1[ f1$V2 == 1,]$V1,
  f1[ f1$V2 == 2,]$V1,
  f1[ f1$V2 == 3,]$V1,          
  f1[ f1$V2 == 4,]$V1,
  names=0:4,
  ylab="f(x)",
  main="distribution of f(x) over classes"
)








