# adience tests!

source("helpers.R")

xent.test.dist = read.csv("../dists/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam.modelv1.65.bak2.test.csv",header=F)
pois.t1.0.xent.test.dist = read.csv("../dists/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.20.bak3.test.csv",header=F)
binom.t1.0.xent.test.dist = read.csv("../dists/adience_binom_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.14.bak3.test.csv",header=F)
pois.t1.0.emd2.test.dist = read.csv("../dists/repeats/adience_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.29.bak3.test.0.csv",header=F)

binom.lt.xent.test.dist = read.csv("../dists/repeats/adience_binom_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.20.bak3.test.0.csv",header=F)

# diabetic tests!

dr.test.dist = read.csv("../dists/repeats/dr_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.30.bak3.test.0.csv",header=F)
dr.pois.t1.0.xent.test.dist = read.csv("../dists/repeats/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.20.bak3.1.test.csv",header=F)
dr.binom.t1.0.xent.test.dist = read.csv("../dists/repeats/dr_binom_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.20.bak3.test.0.csv",header=F)
dr.binom.lt.xent.test.dist = read.csv("../dists/repeats/dr_binom_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.19.bak3.test.0.csv",header=F)

# diabetic on full train+valid
dr.test.dist.full = read.csv("../dists/repeats/dr_xent_l2-1e-4_sgd_pre_split_hdf5_adam_absorb-valid.modelv1.150.bak.0.test.csv",header=F)

dr.binom.lt.xent.test.dist.full = read.csv("../dists/repeats/dr_binom_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam_absorb-valid.modelv1.150.bak.test.0.csv",header=F)

dfemd2.binom.t1.0.adam.test.dist.bestxent = read.csv("../dists/repeats/dr_binom_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.41.bak2.bestxentqwk.test.0.csv",header=F)

dr.test.dist.bestxent = read.csv("../dists/repeats/dr_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.15.bak3.bestxent.0.csv",header=F)

top.m.accuracy = function(preds, k=8, m=1) {
  num.correct = 0
  for(i in 1:nrow(preds)) {
    # compute the top m predictions
    pred.top.m = order( as.numeric(preds[i,1:k]),decreasing=T)[1:m] -1
    # is the ground truth label in these predictions?
    if( preds[i,k+1] %in% pred.top.m ) {
      num.correct = num.correct + 1
    }
  }
  return(num.correct / nrow(preds))
}

# barplots for adience

# x-ent
# -----
aa = c()
for(z in 1:3) { aa = c(aa, top.m.accuracy(xent.test.dist, m=z)) }

# pois + tau=1 + x-ent
# --------------------
bb = c()
for(z in 1:3) { bb = c(bb, top.m.accuracy(pois.t1.0.xent.test.dist, m=z)) }

# binom + tau=1 + x-ent
# ---------------------
cc = c()
for(z in 1:3) { cc = c(cc, top.m.accuracy(binom.t1.0.xent.test.dist, m=z)) }

# binom + tau=learned + x-ent
# ---------------------------
dd = c()
for(z in 1:3) { dd = c(dd, top.m.accuracy(binom.lt.xent.test.dist, m=z)) }

prep.margins = function() {
  par(mar=c(2,2,1,2)+0.1) 
}

pdf("adience_test_set.pdf", width=8, height=4)
prep.margins()
barplot( 
  matrix( c(aa,bb,cc,dd), nrow=4, byrow=T), 
  beside=T, 
  names.arg=c("k=1", "k=2", "k=3"), 
  col=c(preset.cols[1],preset.cols[2],preset.cols[3],preset.cols[4]),
  ylim=c(0,1),
)
legend("topleft", 
       legend=c("x-ent","x-ent/pois/tau=1","x-ent/binom/tau=1","x-ent/binom/tau=L"), 
       fill=c(preset.cols[1],preset.cols[2],preset.cols[3],preset.cols[4]),
       bty="n")
dev.off()

# barplots for DR

aa.dr = c()
for(z in 1:2) { aa.dr = c(aa.dr, top.m.accuracy(dr.test.dist,k=5, m=z)) }

bb.dr = c()
for(z in 1:2) { bb.dr = c(bb.dr, top.m.accuracy(dr.pois.t1.0.xent.test.dist,k=5, m=z)) }

cc.dr = c()
for(z in 1:2) { cc.dr = c(cc.dr, top.m.accuracy(dr.binom.t1.0.xent.test.dist,k=5, m=z)) }

barplot( 
  matrix( c(aa.dr,bb.dr,cc.dr), nrow=3, byrow=T), 
  beside=T, 
  names.arg=c("m=1", "m=2"), 
  col=c(preset.cols[1],preset.cols[2],preset.cols[3]),
  ylim=c(0,1),
)

# ---------------------------------------------
# test different replicates for xent.test.dist
# ---------------------------------------------

xent.test.dists = list(m1=NULL,m2=NULL,m3=NULL)
for(r in 0:4) {
  tmp = read.csv(paste("../dists/repeats/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam.modelv1.65.bak2.test.",r,".csv",sep=""))
  xent.test.dists$m1 = c(xent.test.dists$m1, top.m.accuracy(tmp, m=1))
  xent.test.dists$m2 = c(xent.test.dists$m2, top.m.accuracy(tmp, m=2))
  xent.test.dists$m3 = c(xent.test.dists$m3, top.m.accuracy(tmp, m=3))
}

# ------------------------------------------------------
# test different replicates for pois.t1.0.xent.test.dist
# ------------------------------------------------------

pois.t1.0.xent.test.dists = list(m1=NULL, m2=NULL, m3=NULL)
for(r in 0:4) {
  tmp = read.csv(paste("../dists/repeats/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.20.bak3.test.",r,".csv",sep=""))
  pois.t1.0.xent.test.dists$m1 = c(pois.t1.0.xent.test.dists$m1, top.m.accuracy(tmp, m=1))
  pois.t1.0.xent.test.dists$m2 = c(pois.t1.0.xent.test.dists$m2, top.m.accuracy(tmp, m=2))
  pois.t1.0.xent.test.dists$m3 = c(pois.t1.0.xent.test.dists$m3, top.m.accuracy(tmp, m=3))
}

# --------------------------------------------------------
# test different replicates for binom.t1.0.xent.test.dists
# --------------------------------------------------------

binom.t1.0.xent.test.dists = list(m1=NULL, m2=NULL, m3=NULL)
for(r in 0:4) {
  tmp = read.csv(paste("../dists/repeats/adience_binom_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam.modelv1.14.bak3.test.",r,".csv",sep=""))
  binom.t1.0.xent.test.dists$m1 = c(binom.t1.0.xent.test.dists$m1, top.m.accuracy(tmp, m=1))
  binom.t1.0.xent.test.dists$m2 = c(binom.t1.0.xent.test.dists$m2, top.m.accuracy(tmp, m=2))
  binom.t1.0.xent.test.dists$m3 = c(binom.t1.0.xent.test.dists$m3, top.m.accuracy(tmp, m=3))
}

