source("helpers.R")

afx =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")[1:100,]

afemd2.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afemd2.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.3_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afx.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_tau-0.3_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afx.pois.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afemd2.pois.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afx.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_tau-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")

afx.pois.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afx.pois.t0.5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afx.pois.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.3_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afx.pois.t0.1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.1_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afx.pois.scap.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_scap_t-1_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")

legnd = function(where="topright") {
  legend(where, 
         legend=c("x-ent", "x-ent + pois (tau = 1)", "x-ent + pois (tau = 0.5)", "x-ent + pois (tau = 0.3)"), 
         col=c("black","red","orange","purple"), lty="solid", lwd=2, cex=0.5)
}

par(mfrow=c(1,1))

plot(afx$valid_xent,type="l",xlim=c(0,100), lwd=1.5, xlab="epoch", ylab="valid cross-entropy")
lines(afx.p1$valid_xent,col="red", lwd=1.5)
lines(afe.p1$valid_xent,col="blue", lwd=1.5)
lines(afx.p5$valid_xent,col="orange", lwd=1.5)
lines(afx.p3$valid_xent,col="purple", lwd=1.5)
legnd("topright")

plot(afx$valid_xent_accuracy,type="l",xlim=c(0,110), ylim=c(0.1,1.0), lwd=1.5, xlab="epoch", ylab="valid accuracy")
lines(afx.pois.t1.0$valid_xent_accuracy,col="red", lwd=1.5)
lines(afemd2.pois.t1.0$valid_xent_accuracy,col="blue", lwd=1.5)
lines(afx.pois.t0.5$valid_xent_accuracy,col="orange", lwd=1.5)
lines(afx.pois.t0.3$valid_xent_accuracy,col="purple", lwd=1.5)
lines(afx.pois.t0.1$valid_xent_accuracy,col="brown", lwd=1.5)
#lines(afx.pois.scap.t1.0$valid_xent_accuracy,col="green", lwd=1.5); # no workie
lines(afx.lt$valid_xent_accuracy,col="green", lwd=1.5);
legnd("bottomright")

plot(afx$valid_xent_qwk,type="l",xlim=c(0,100), ylim=c(0.1,1.0), lwd=1.5, xlab="epoch", ylab="valid qwk")
lines(afx.pois.t1.0$valid_xent_qwk,col="red", lwd=1.5)
lines(afemd2.pois.t1.0$valid_xent_qwk,col="blue", lwd=1.5)
lines(afx.pois.t0.5$valid_xent_qwk,col="orange", lwd=1.5)
lines(afx.pois.t0.3$valid_xent_qwk,col="purple", lwd=1.5)
lines(afx.lt$valid_xent_qwk,col="green", lwd=1.5)
legnd("bottomright")


# ----------------

# ----------------

# x-ent (tau=1) vs x-ent/pois (tau=1) vs emd2/pois (tau=1)

par(mfrow=c(1,1))

ad.curves = function(coln, ylab, legend.where) {
  plot(get.loess(afx[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(afx[coln][,1],col=preset.cols.alpha[1])
  lines(get.loess(afx.pois.t1.0[coln][,1]),col=preset.cols[2], lwd=1.5); lines(afx.pois.t1.0[coln][,1],col=preset.cols.alpha[2])
  lines(get.loess(afemd2.t1.0[coln][,1]),col=preset.cols[3], lwd=1.5); lines(afemd2.t1.0[coln][,1],col=preset.cols.alpha[3])
  legend(legend.where, 
         legend=c("x-ent (tau=1)","x-ent + pois (tau=1)", "emd2 + pois (tau=1)"),
         col=preset.cols[1:3],lty="solid",lwd=1.5,bty="n",cex=0.5)
}

pdf("ad_curves.pdf", width=fig.width, height=fig.height)

par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)

ad.curves("valid_xent", "valid x-ent", "topright")
ad.curves("valid_xent_accuracy", "valid acc", "bottomright")
ad.curves("valid_xent_qwk", "valid qwk", "bottomright")

dev.off()


# ----------

# x-ent (tau=1) vs x-ent/pois (tau=0.3) vs emd2/pois (tau=0.3)

ad.curves.2 = function(coln, ylab, legend.where) {
  #plot(get.loess(afx.t0.3[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(afx.t0.3[coln][,1],col=preset.cols.alpha[1])
  plot(get.loess(afx[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(afx[coln][,1],col=preset.cols.alpha[1]) 
  lines(get.loess(afx.pois.t0.3[coln][,1]),col=preset.cols[2], lwd=1.5); lines(afx.pois.t0.3[coln][,1],col=preset.cols.alpha[2])
  lines(get.loess(afemd2.t0.3[coln][,1]),col=preset.cols[3], lwd=1.5); lines(afemd2.t0.3[coln][,1],col=preset.cols.alpha[3])
  legend(legend.where, 
         legend=c("x-ent","x-ent + pois (tau=0.3)", "emd2 + pois (tau=0.3)"),
         col=preset.cols[1:4],lty="solid",lwd=1.5,bty="n",cex=0.5)
}

pdf("ad_curves_2.pdf", width=fig.width, height=fig.height)

par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)

ad.curves.2("valid_xent", "valid x-ent", "topright")
ad.curves.2("valid_xent_accuracy", "valid acc", "bottomright")
ad.curves.2("valid_xent_qwk", "valid qwk", "bottomright")

dev.off()


# ----------

# x-ent (tau=1.0) vs x-ent/pois (tau=learn) vs emd2/pois (tau=learn)

ad.curves.3 = function(coln, ylab, legend.where) {
  #plot(get.loess(afx.pois.lt[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(afx.pois.lt[coln][,1],col=preset.cols.alpha[1])
  plot(get.loess(afx[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(afx[coln][,1],col=preset.cols.alpha[1]) 
  lines(get.loess(afx.pois.lt[coln][,1]),col=preset.cols[2], lwd=1.5); lines(afx.pois.lt[coln][,1],col=preset.cols.alpha[2])
  lines(get.loess(afemd2.pois.lt[coln][,1]),col=preset.cols[3], lwd=1.5); lines(afemd2.pois.lt[coln][,1],col=preset.cols.alpha[3])
  legend(legend.where, 
         legend=c("x-ent (tau=1)","x-ent + pois (tau=learn)", "emd2 + pois (tau=learn)"),
         col=preset.cols[1:4],lty="solid",lwd=1.5,bty="n",cex=0.5)
}

pdf("ad_curves_3.pdf", width=fig.width, height=fig.height)

par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)

ad.curves.3("valid_xent", "valid x-ent", "topright")
ad.curves.3("valid_xent_accuracy", "valid acc", "bottomright")
ad.curves.3("valid_xent_qwk", "valid qwk", "bottomright")

dev.off()


