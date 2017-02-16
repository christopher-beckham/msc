source("helpers.R")

dfx =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
dfx.pois.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_dlra100//results.txt")

dfx.t0.3 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_tau-0.3_l2-1e-4_sgd_pre_split_hdf5//results.txt")
dfx.pois.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.3_xent_l2-1e-4_sgd_pre_split_hdf5_dlra100/results.txt")

dfemd2.pois.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

dfx.pois.t0.5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")

# ------------------------------------

legnd = function(where="topright") {
  legend(where, legend=c("x-ent (tau = 1)", "x-ent + pois (tau = 1)", "x-ent + pois (tau = 0.3)"), col=c("black","red","purple"), lty="solid", lwd=2, cex=0.5)
}

pdf("dr_curves.pdf", width=6, height=4)

par(mfrow=c(1,2))

plot(dfx$valid_xent,type="l",xlim=c(0,150), ylim=c(0.6,1.0), lwd=1.5, xlab="epoch", ylab="valid cross-entropy")
lines(dfx.p1$valid_xent,col="red", lwd=1.5)
lines(dfx.p5$valid_xent,col="orange", lwd=1.5)
lines(dfx.p3$valid_xent,col="purple", lwd=1.5)
legnd("topright")

plot(dfx$valid_xent_qwk,type="l",xlim=c(0,150), ylim=c(0.4,0.75), lwd=1.5, xlab="epoch", ylab="valid qwk")
lines(dfx.p1$valid_xent_qwk,col="red", lwd=1.5)
lines(dfx.p5$valid_xent_qwk,col="orange", lwd=1.5)
lines(dfx.p3$valid_xent_qwk,col="purple", lwd=1.5)
legnd("bottomright")

dev.off()

# ------------------------------------

pdf("dr_curves_2.pdf", width=6, height=2)

par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)
labels = c("x-ent (tau=1)", "x-ent/pois (tau=1)", "emd2/pois (tau=1)")
# x-ent (tau=1) vs x-ent/pois (tau=1) vs emd2/pois (tau=1)
dr.curves = function(coln, legend.where) {
  plot(get.loess(dfx[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab="valid x-ent"); lines(dfx[coln][,1],col=preset.cols.alpha[1])
  lines(get.loess(dfx.pois.t1.0[coln][,1]),col=preset.cols[2], lwd=1.5); lines(dfx.pois.t1.0[coln][,1],col=preset.cols.alpha[2])
  lines(get.loess(dfemd2.pois.t1.0[coln][,1]),col=preset.cols[3], lwd=1.5); lines(dfemd2.pois.t1.0[coln][,1],col=preset.cols.alpha[3])
  legend(legend.where, legend=labels, lwd=1.5, col=preset.cols[1:3], cex=0.5)  
}
dr.curves("valid_xent", "topright")
dr.curves("valid_xent_accuracy", "bottomright")
dr.curves("valid_xent_qwk", "bottomright")
dev.off()



pdf("dr_curves_2b.pdf", width=6, height=2)
par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)

# x-ent (tau=0.3) vs pois (tau=0.3)

dr.curves.2 = function(coln, legend.where) {
  plot(get.loess(dfx.t0.3[coln][,1]),col=preset.cols[1], xlim=c(0,150), lwd=1.5, type="l", xlab="epoch", ylab="valid x-ent"); lines(dfx.t0.3[coln][,1],col=preset.cols.alpha[1])
  lines(get.loess(dfx.pois.t0.3[coln][,1]),col=preset.cols[2], lwd=1.5); lines(dfx.pois.t0.3[coln][,1],col=preset.cols.alpha[2])
  legend(legend.where, legend=labels, lwd=1.5, col=preset.cols[1:3], cex=0.5)  
}

dr.curves.2("valid_xent","topright")
dr.curves.2("valid_xent_accuracy","bottomright")
dr.curves.2("valid_xent_qwk","bottomright")

dev.off()

# ------------------------------------


par(mfrow=c(1,1))

plot(dfx$valid_xent_accuracy,type="l",xlim=c(0,150), lwd=1.5, xlab="epoch", ylab="valid cross-entropy")
lines(dfx.p1$valid_xent_accuracy,col="red", lwd=1.5)
lines(dfx.p3$valid_xent_accuracy,col="purple", lwd=1.5)
