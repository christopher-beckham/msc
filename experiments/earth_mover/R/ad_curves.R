source("helpers.R")

afx =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")[1:100,]
afx.adam =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam//results.txt")

afx.binom = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-1_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")
afx.binom.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam//results.txt")

afemd2.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afemd2.t1.0.sched = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_sched/results.txt")

afemd2.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.3_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afx.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_tau-0.3_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afx.pois.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afx.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
afemd2.pois.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afemd2.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

afx.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_tau-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")

afx.pois.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afx.pois.t1.0.sched = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_sched//results.txt")

afx.pois.t0.5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afx.pois.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.3_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
afx.pois.t0.1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.1_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")

afx.pois.scap.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_scap_t-1_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")

af.sq = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_sq_l2-1e-4_sgd_pre_split_hdf5/results.txt")
af.sqclassic =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_sqclassic_l2-1e-4_sgd_pre_split_hdf5_lr01/results.txt")
#af.sq.relu =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_sq


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

ad.curves = function(colns, ylab, legend.where) {
  plot(get.loess(afx[colns[1]][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(afx[colns[1]][,1],col=preset.cols.alpha[1])
  lines(get.loess(afx.pois.t1.0[colns[2]][,1]),col=preset.cols[2], lwd=1.5); lines(afx.pois.t1.0[colns[2]][,1],col=preset.cols.alpha[2])
  lines(get.loess(afemd2.t1.0[colns[3]][,1]),col=preset.cols[3], lwd=1.5); lines(afemd2.t1.0[colns[3]][,1],col=preset.cols.alpha[3])
  lines(get.loess(af.sq[colns[4]][,1]),col=preset.cols[4], lwd=1.5); lines(af.sq[colns[4]][,1],col=preset.cols.alpha[4])
  legend(legend.where, 
         legend=c("x-ent (tau=1)","x-ent + pois (tau=1)", "emd2 + pois (tau=1)"),
         col=preset.cols[1:4],lty="solid",lwd=1.5,bty="n",cex=0.5)
}

pdf("ad_curves.pdf", width=fig.width, height=fig.height)

par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)

ad.curves(rep("valid_xent",4), "valid x-ent", "topright")
ad.curves(c("valid_xent_accuracy","valid_xent_accuracy","valid_xent_accuracy","valid_exp_accuracy"), "valid acc", "bottomright")
ad.curves(c("valid_xent_qwk","valid_xent_qwk","valid_xent_qwk","valid_exp_qwk"), "valid qwk", "bottomright")

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

# --------

tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsigfn_xent_l2-1e-4_sgd_pre_split_hdf5_repeat/results.txt")
tmp2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-0.125_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")


par(mfrow=c(1,1))

plot(afx$valid_xent,type="l")
lines(tmp$valid_xent,col="red")

plot(afx$valid_xent_accuracy,type="l",xlim=c(0,200))
lines(afx.pois.t0.3$valid_xent_accuracy,col="brown")
lines(tmp2$valid_xent_accuracy,col="green")
lines(tmp$valid_xent_accuracy,col="red")

plot(afx$valid_xent_qwk,type="l",xlim=c(0,200))
lines(tmp2$valid_xent_qwk,col="green")
lines(tmp$valid_xent_qwk,col="red")


plot(afx$valid_exp_accuracy,type="l",xlim=c(0,200))
lines(afx.pois.t0.3$valid_exp_accuracy,col="brown")
lines(tmp2$valid_exp_accuracy,col="green")
lines(tmp$valid_exp_accuracy,col="red")


# -------

plot(afx$valid_exp_accuracy,type="l"); lines(afx.pois.lt.adam$valid_exp_accuracy,col="red")

plot(afx$valid_exp_qwk,type="l"); lines(afx.pois.lt.adam$valid_exp_qwk,col="red")

plot(afx$valid_exp_accuracy,type="l"); lines(afemd2.pois.lt.adam$valid_exp_accuracy,col="red")

plot(afx$valid_exp_qwk,type="l"); lines(afemd2.pois.lt.adam$valid_exp_qwk,col="red")


afx.pois.lfn.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsigfn_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

plot(afx.adam$valid_exp_accuracy,type="l",xlim=c(0,100), ylim=c(0.4,0.9))
lines(afx.pois.lt.adam$valid_exp_accuracy,col="red")
lines(afemd2.pois.lt.adam$valid_exp_accuracy,col="blue")
lines(afx.binom.adam$valid_exp_accuracy,col="orange")
lines(afx.pois.lfn.adam$valid_exp_accuracy,col="brown")
legend("bottomright", 
       c("x-ent+adam", "pois+lt+adam", "emd+pois+lt+adam","binom+adam","pois+lfn+adam"), 
       col=c("black","red","blue","orange","brown"), lty="solid")

plot(afx.adam$valid_exp_qwk,type="l",xlim=c(0,100))
lines(afx.pois.lt.adam$valid_exp_qwk,col="red")
lines(afemd2.pois.lt.adam$valid_exp_qwk,col="blue")

plot(afx.adam$valid_loss,type="l")

