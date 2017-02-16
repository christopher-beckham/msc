# -------------
# look at f(x)
# -------------

library(reshape2)
library(ggplot2)
library(xtables)

f1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.200.bak.fx.csv",header=F)

par(mfrow=c(1,1))

plot(density(f1[ f1$V2 == 0,]$V1),ylim=c(0,1))
lines(density(f1[ f1$V2 == 1,]$V1),col="red")
lines(density(f1[ f1$V2 == 2,]$V1),col="orange")
lines(density(f1[ f1$V2 == 3,]$V1),col="blue")
lines(density(f1[ f1$V2 == 4,]$V1),col="purple")

classes = c("no DR", "mild DR", "modt. DR", "severe DR", "prolif. DR")

boxplot(
  f1[ f1$V2 == 0,]$V1,
  f1[ f1$V2 == 1,]$V1,
  f1[ f1$V2 == 2,]$V1,
  f1[ f1$V2 == 3,]$V1,          
  f1[ f1$V2 == 4,]$V1,
  names=classes,
  ylab="f(x)",
  las=2,
  main="diabetic retinopathy (tau = 1)"
)

flist=list(
  c0=f1[ f1$V2 == 0,]$V1, 
  c1=f1[ f1$V2 == 1,]$V1,
  c2=f1[ f1$V2 == 2,]$V1,
  c3=f1[ f1$V2 == 3,]$V1,
  c4=f1[ f1$V2 == 4,]$V1
)
means = sapply(flist, mean)

dist(means,diag=T)

tmp = as.matrix(dist(means,diag=T))
tmp = round(tmp, 2)
tmp[ upper.tri(tmp) ] = " "
rownames(tmp) = classes
colnames(tmp) = classes

xtable(tmp)


# -----------------

f2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.100.bak.fx.csv",header=F)

classes2 = c("0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+")
boxplot(
  f2[ f2$V2 == 0,]$V1,
  f2[ f2$V2 == 1,]$V1,
  f2[ f2$V2 == 2,]$V1,
  f2[ f2$V2 == 3,]$V1,          
  f2[ f2$V2 == 4,]$V1,
  f2[ f2$V2 == 5,]$V1,          
  f2[ f2$V2 == 6,]$V1,
  f2[ f2$V2 == 7,]$V1,
  names=classes2,
  las=2,
  ylim=c(0,10),
  ylab="f(x)",
  main="adience (tau = 1)"
)

flist=list(
  c0=f2[ f2$V2 == 0,]$V1, 
  c1=f2[ f2$V2 == 1,]$V1,
  c2=f2[ f2$V2 == 2,]$V1,
  c3=f2[ f2$V2 == 3,]$V1,
  c4=f2[ f2$V2 == 4,]$V1,
  c5=f2[ f2$V2 == 5,]$V1,
  c6=f2[ f2$V2 == 6,]$V1,
  c7=f2[ f2$V2 == 7,]$V1
)
means = sapply(flist, mean)

tmp = as.matrix(dist(means,diag=T))
tmp = round(tmp, 2)
tmp[ upper.tri(tmp) ] = " "
rownames(tmp) = classes2
colnames(tmp) = classes2

xtable(tmp)


# ----

pdf("boxplots.pdf",height=4, width=7)
par(mfrow=c(1,2))
boxplot(
  f1[ f1$V2 == 0,]$V1,
  f1[ f1$V2 == 1,]$V1,
  f1[ f1$V2 == 2,]$V1,
  f1[ f1$V2 == 3,]$V1,          
  f1[ f1$V2 == 4,]$V1,
  names=classes,
  ylab="f(x)",
  las=2,
  main="diabetic retinopathy (tau = 1)"
)
boxplot(
  f2[ f2$V2 == 0,]$V1,
  f2[ f2$V2 == 1,]$V1,
  f2[ f2$V2 == 2,]$V1,
  f2[ f2$V2 == 3,]$V1,          
  f2[ f2$V2 == 4,]$V1,
  f2[ f2$V2 == 5,]$V1,          
  f2[ f2$V2 == 6,]$V1,
  f2[ f2$V2 == 7,]$V1,
  names=classes2,
  las=2,
  ylim=c(0,10),
  ylab="f(x)",
  main="adience (tau = 1)"
)
dev.off()



