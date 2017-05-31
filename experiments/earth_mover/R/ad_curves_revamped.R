source("helpers.R")

fig.width = 12
fig.height = 2.5

do.plot = function(dfs, exp.names, exp.cols, exp.ltys, which.col, xlab, ylab, xlim, ylim) {
  for( i in 1:length(dfs)) {
    df = dfs[[i]]
    if(i==1) {
      plot(get.loess(df[which.col][,1]), type="l", lwd=1.5, col=exp.cols[i], lty=exp.ltys[i], xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab)
    } else {
      lines(get.loess(df[which.col][,1]), type="l", lwd=1.5, col=exp.cols[i], lty=exp.ltys[i], xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab)
    }
  }
  legend("bottomright", legend=exp.names, lwd=1.5, col=exp.cols, lty=exp.ltys, bty="n", cex=0.8)
}


# -------------
# adience tau=1
# -------------
# x-ent
afx.adam =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam//results.txt")
afx.adam.s2 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s2/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam//results.txt")
afx.adam.plateau = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam_plateau/results.txt") 
# x-ent + binom + tau1
afx.binom.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam//results.txt")
# x-ent + pois + tau1
afx.pois.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")[1:100,]
afx.pois.t1.0.adam.s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s2/adience_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

# emd2 + pois + tau1
afemd2.pois.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt") ##
# emd2 + binom + tau1
afemd2.binom.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# sq-classic
afsqclassic.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_sqclassic-backrelu_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")

# TAU=1 plots
pdf("adience_tau1_camera-ready.pdf", width=fig.width, height=fig.height)
par(mar=c(5,4,1,1)+0.1, mfrow=c(1,4))
exp.names = c("x-ent (tau=1)", # blue
              "x-ent + pois (tau=1)", # orange
              "emd2 + pois (tau=1)", # orange + dashed
              "x-ent + binom (tau=1)", # green
              "emd2 + binom (tau=1)", # green + dashed
              "sq-err") # red
exp.cols = c(preset.cols[1], preset.cols[2], preset.cols[2], preset.cols[3], preset.cols[3], preset.cols[4])
exp.ltys = c("solid", "solid", "twodash", "solid", "twodash", "solid")
ad.tau1.dfs = list(afx.adam, afx.pois.t1.0.adam, afemd2.pois.t1.0.adam, afx.binom.t1.0.adam, afemd2.binom.t1.0.adam, afsqclassic.adam)
do.plot(ad.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_accuracy", "epoch", "valid acc (argmax)", c(0,100), c(0.3, 0.85))
do.plot(ad.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_accuracy", "epoch", "valid acc (exp)", c(0,100), c(0.3, 0.85))
do.plot(ad.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_qwk", "epoch", "valid QWK (argmax)", c(0,100), c(0.8, 0.96))
do.plot(ad.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_qwk", "epoch", "valid QWP (exp)", c(0,100), c(0.8, 0.96))
dev.off()


# -------------------
# adience tau=learned
# -------------------
# learn tau for x-ent and emd2 poisson
afx.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")[1:100,]
afx.pois.lt.adam.s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s2/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

afemd2.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# learn tau for x-ent and emd2 binom
afx.binom.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt") ##
afemd2.binom.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# random stuff
afx.pois.lfn.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsigfn-fixed_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

afx.binom.lfn.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/")


# TAU=learned plots
pdf("adience_taulearn_camera-ready.pdf", width=fig.width, height=fig.height)
par(mar=c(5,4,1,1)+0.1, mfrow=c(1,4))
exp.names = c("x-ent (tau=1)",
              "x-ent + pois (tau=learned)", 
              "emd2 + pois (tau=learned)",
              "x-ent + binom (tau=learned)",
              "emd2 + binom (tau=learned)",
              "sq-err")
exp.cols = c(preset.cols[1], preset.cols[2], preset.cols[2], preset.cols[3], preset.cols[3], preset.cols[4])
exp.ltys = c("solid", "solid", "twodash", "solid", "twodash", "solid")
ad.lt.dfs = list(afx.adam, afx.pois.lt.adam, afemd2.pois.lt.adam, afx.binom.lt.adam, afemd2.binom.lt.adam, afsqclassic.adam)
do.plot(ad.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_accuracy", "epoch", "valid acc (argmax)", c(0,100), c(0.3, 0.85))
do.plot(ad.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_accuracy", "epoch", "valid acc (exp)", c(0,100), c(0.3, 0.85))
do.plot(ad.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_qwk", "epoch", "valid QWK (argmax)", c(0,100), c(0.8, 0.96))
do.plot(ad.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_qwk", "epoch", "valid QWP (exp)", c(0,100), c(0.8, 0.96))
dev.off()


