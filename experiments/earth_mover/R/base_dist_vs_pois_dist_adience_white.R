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

num.classes = 8

pdf("713_xent_vs_pois.pdf",height=4)
par(mfrow=c(1,2))
cls.col = rep("#1f77b4", 8)
cls.col[ adx.dist1[713,]$V9+1 ] = "#ff7f03"
barplot(as.numeric(adx.dist1[713,1:num.classes]), names.arg=0:(num.classes-1),
        col.lab="white",col.axis="white",col.main="white",col.axis="white",col=cls.col, main="baseline") 
barplot(as.numeric(adxp.dist1[713,1:num.classes]), names.arg=0:(num.classes-1),
        col.lab="white",col.axis="white",col.main="white",col.axis="white", col=cls.col, main="poisson")
dev.off()


pdf("223_xent_vs_pois.pdf",height=4)
par(mfrow=c(1,2))
cls.col = rep("#1f77b4", 8)
cls.col[ adx.dist1[223,]$V9+1 ] = "#ff7f03"
barplot(as.numeric(adx.dist1[223,1:num.classes]), names.arg=0:(num.classes-1),
        col.lab="white",col.axis="white",col.main="white",col.axis="white",col=cls.col, main="baseline") 
barplot(as.numeric(adxp.dist1[223,1:num.classes]), names.arg=0:(num.classes-1),
        col.lab="white",col.axis="white",col.main="white",col.axis="white", col=cls.col, main="poisson")
dev.off()

#axes=F only deals with the vertical

pdf("911_xent_vs_pois.pdf",height=4)
par(mfrow=c(1,2))
cls.col = rep("#1f77b4", 8)
cls.col[ adx.dist1[911,]$V9+1 ] = "#ff7f03"
barplot(as.numeric(adx.dist1[911,1:num.classes]), names.arg=0:(num.classes-1),
        col.lab="white",col.axis="white",col.main="white",col=cls.col, main="baseline")
barplot(as.numeric(adxp.dist1[911,1:num.classes]), names.arg=0:(num.classes-1),
        col.lab="white",col.axis="white",col.main="white",col=cls.col, main="poisson")
dev.off()

