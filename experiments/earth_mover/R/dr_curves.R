source("helpers.R")

dfx =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
dfx.pois.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_dlra100//results.txt")

dfx.t0.3 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_tau-0.3_l2-1e-4_sgd_pre_split_hdf5_dlra100//results.txt")
dfx.pois.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.3_xent_l2-1e-4_sgd_pre_split_hdf5_dlra100/results.txt")

dfemd2.pois.t1.0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_dlra100/results.txt")

dfx.pois.t0.5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.5_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")

dfemd2.pois.t0.3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-0.3_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

dfx.pois.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
dfemd2.pois.lt = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")

# ------------------------------------

pdf("dr_curves.pdf", width=fig.width, height=fig.height)

par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)
labels = c("x-ent (tau=1)", "x-ent/pois (tau=1)", "emd2/pois (tau=1)")
# x-ent (tau=1) vs x-ent/pois (tau=1) vs emd2/pois (tau=1)
dr.curves = function(coln, ylab, legend.where) {
  plot(get.loess(dfx[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(dfx[coln][,1],col=preset.cols.alpha[1])
  lines(get.loess(dfx.pois.t1.0[coln][,1]),col=preset.cols[2], lwd=1.5); lines(dfx.pois.t1.0[coln][,1],col=preset.cols.alpha[2])
  lines(get.loess(dfemd2.pois.t1.0[coln][,1]),col=preset.cols[3], lwd=1.5); lines(dfemd2.pois.t1.0[coln][,1],col=preset.cols.alpha[3])
  legend(legend.where, legend=c("x-ent (tau = 1)", "x-ent + pois (tau = 1)", "x-ent + pois (tau = 1)"), col=preset.cols[1:3], lty="solid", lwd=1.5, cex=0.5)
}
dr.curves("valid_xent", "valid x-ent","topright")
dr.curves("valid_xent_accuracy", "valid acc","bottomright")
dr.curves("valid_xent_qwk", "valid qwk","bottomright")
dev.off()

pdf("dr_curves_2.pdf", width=fig.width, height=fig.height)
par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)

# x-ent (tau=1.0) vs x-ent/pois (tau=0.3) vs emd2/pois 

dr.curves.2 = function(coln, ylab, legend.where, ylim=NULL) {
  #plot(get.loess(dfx.t0.3[coln][,1]),col=preset.cols[1], xlim=c(0,150), lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(dfx.t0.3[coln][,1],col=preset.cols.alpha[1])
  plot(get.loess(dfx[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylim=ylim, ylab=ylab); lines(dfx[coln][,1],col=preset.cols.alpha[1])
  lines(get.loess(dfx.pois.t0.3[coln][,1]),col=preset.cols[2], lwd=1.5); lines(dfx.pois.t0.3[coln][,1],col=preset.cols.alpha[2])
  #lines(get.loess(dfemd2.pois.t0.3[coln][,1]),col=preset.cols[3], lwd=1.5); lines(dfemd2.pois.t0.3[coln][,1],col=preset.cols.alpha[3])
  legend(legend.where, legend=c("x-ent (tau = 1)", "x-ent + pois (tau = 0.3)", "x-ent + pois (tau = 0.3)"), col=preset.cols[1:3], lty="solid", lwd=1.5, cex=0.5)  
}

dr.curves.2("valid_xent","valid x-ent","topright")
dr.curves.2("valid_xent_accuracy","valid acc","bottomright")
dr.curves.2("valid_xent_qwk","valid qwk","bottomright",ylim=c(0.1,0.7))

dev.off()

# ------------------------------------

pdf("dr_curves_3.pdf", width=6, height=2)
par(mfrow=c(1,3))
par(mar=c(5,4,1,1)+0.1)

dr.curves.3= function(coln, ylab, legend.where) {
  plot(get.loess(dfx[coln][,1]),col=preset.cols[1], lwd=1.5, type="l", xlab="epoch", ylab=ylab); lines(dfx[coln][,1],col=preset.cols.alpha[1])
  lines(get.loess(dfx.pois.lt[coln][,1]),col=preset.cols[2], lwd=1.5); lines(dfx.pois.lt[coln][,1],col=preset.cols.alpha[2])
  lines(get.loess(dfemd2.pois.lt[coln][,1]),col=preset.cols[3], lwd=1.5); lines(dfemd2.pois.lt[coln][,1],col=preset.cols.alpha[3])
  legend(legend.where, legend=c("x-ent (tau = 1)", "x-ent + pois (tau = learn)", "x-ent + pois (tau = learn)"), col=preset.cols[1:3], lty="solid", lwd=1.5, cex=0.5)  
}

dr.curves.3("valid_xent","valid x-ent","topright")
dr.curves.3("valid_xent_accuracy","valid acc","bottomright")
dr.curves.3("valid_xent_qwk","valid qwk","bottomright")

dev.off()

