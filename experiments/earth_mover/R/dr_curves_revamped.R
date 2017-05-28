source("helpers.R")

fig.width = 12
fig.height = 2.5

# -------------
# adience tau=1
# -------------
# x-ent
afx.adam =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam//results.txt")
afx.adam.s2 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s2/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam//results.txt")
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
exp.names = c("x-ent (tau=1)",
              "x-ent + pois (tau=1)",
              "emd2 + pois (tau=1)",
              "x-ent + binom (tau=1)",
              "emd2 + binom (tau=1)",
              "sq-err")
# valid x-ent accuracy
plot(get.loess(afx.adam$valid_xent_accuracy),type="l",lwd=1.5,col=preset.cols[1], xlim=c(0,100), ylim=c(0.2, 0.85), xlab="epoch", ylab="valid acc (argmax)")
lines(get.loess(afx.pois.t1.0.adam$valid_xent_accuracy),lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.t1.0.adam$valid_xent_accuracy),lwd=1.5, col=preset.cols[3]) #FIX
lines(get.loess(afx.binom.t1.0.adam$valid_xent_accuracy),lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.t1.0.adam$valid_xent_accuracy),lwd=1.5,col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_xent_accuracy),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names[1:6], lwd=1.5, col=preset.cols[1:6], bty="n", cex=0.8)
# valid exp accuracy
plot(get.loess(afx.adam$valid_exp_accuracy),type="l",lwd=1.5, col=preset.cols[1], xlim=c(0,100), ylim=c(0.3,0.85), xlab="epoch", ylab="valid acc (exp)")
lines(get.loess(afx.pois.t1.0.adam$valid_exp_accuracy), lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.t1.0.adam$valid_exp_accuracy), lwd=1.5, col=preset.cols[3]) #FIX
lines(get.loess(afx.binom.t1.0.adam$valid_exp_accuracy), lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.t1.0.adam$valid_exp_accuracy),lwd=1.5,col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_exp_accuracy),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names[1:6], lwd=1.5, col=preset.cols[1:6], bty="n", cex=0.8)
# valid xent qwk
plot(get.loess(afx.adam$valid_xent_qwk),type="l", lwd=1.5, col=preset.cols[1], xlim=c(0,100), ylim=c(0.8,0.95), xlab="epoch", ylab="valid QWK (argmax)")
lines(get.loess(afx.pois.t1.0.adam$valid_xent_qwk), lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.t1.0.adam$valid_xent_qwk), lwd=1.5, col=preset.cols[3]) #FIX
lines(get.loess(afx.binom.t1.0.adam$valid_xent_qwk), lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.t1.0.adam$valid_xent_qwk),lwd=1.5,col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_xent_qwk),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names[1:6], lwd=1.5, col=preset.cols[1:6], bty="n", cex=0.8)
# valid exp qwk
plot(get.loess(afx.adam$valid_exp_qwk),type="l", lwd=1.5, col=preset.cols[1], xlim=c(0,100), ylim=c(0.8, 0.95), xlab="epoch", ylab="valid QWK (exp)")
lines(get.loess(afx.pois.t1.0.adam$valid_exp_qwk), lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.t1.0.adam$valid_exp_qwk), lwd=1.5, col=preset.cols[3]) #FIX
lines(get.loess(afx.binom.t1.0.adam$valid_exp_qwk), lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.t1.0.adam$valid_exp_qwk),lwd=1.5,col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_exp_qwk),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names[1:6], lwd=1.5, col=preset.cols[1:6], bty="n", cex=0.8)
dev.off()

# -------------------
# adience tau=learned
# -------------------
# learn tau for x-ent and emd2 poisson
afx.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")[1:100,]
afemd2.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# learn tau for x-ent and emd2 binom
afx.binom.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt") ##
afemd2.binom.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_binom_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# random stuff
afx.pois.lfn.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsigfn-fixed_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

afx.pois.ltfn.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_pois_t-learnsigfn-fixed_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

# TAU=learned plots
pdf("adience_taulearn_camera-ready.pdf", width=fig.width, height=fig.height)
par(mar=c(5,4,1,1)+0.1, mfrow=c(1,4))
exp.names = c("x-ent (tau=1)",
              "x-ent + pois (tau=learned)", 
              "emd2 + pois (tau=learned)",
              "x-ent + binom (tau=learned)",
              "emd2 + binom (tau=learned)",
              "sq-err")
# valid x-ent accuracy
plot(get.loess(afx.adam$valid_xent_accuracy), type="l", lwd=1.5, col=preset.cols[1], xlim=c(0,100), ylim=c(0.3, 0.85), xlab="epoch", ylab="valid acc (argmax)")
lines(get.loess(afx.pois.lt.adam$valid_xent_accuracy), lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.lt.adam$valid_xent_accuracy), lwd=1.5, col=preset.cols[3])
lines(get.loess(afx.binom.lt.adam$valid_xent_accuracy), lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.lt.adam$valid_xent_accuracy), lwd=1.5, col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_xent_accuracy),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names, lwd=1.5, col=preset.cols, bty="n", cex=0.8)
# valid exp accuracy
plot(get.loess(afx.adam$valid_exp_accuracy),type="l", col=preset.cols[1], xlim=c(0,100), ylim=c(0.3, 0.85), xlab="epoch", ylab="valid acc (exp)")
lines(get.loess(afx.pois.lt.adam$valid_exp_accuracy),lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.lt.adam$valid_exp_accuracy),lwd=1.5, col=preset.cols[3])
lines(get.loess(afx.binom.lt.adam$valid_exp_accuracy), lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.lt.adam$valid_exp_accuracy), lwd=1.5, col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_exp_accuracy),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names, lwd=1.5, col=preset.cols, bty="n", cex=0.8)
# valid x-ent qwk
plot(get.loess(afx.adam$valid_xent_qwk),type="l", lwd=1.5, col=preset.cols[1], xlab="epoch", ylab="valid QWK (exp)", ylim=c(0.8,0.96))
lines(get.loess(afx.pois.lt.adam$valid_xent_qwk), lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.lt.adam$valid_xent_qwk), lwd=1.5, col=preset.cols[3])
lines(get.loess(afx.binom.lt.adam$valid_xent_qwk), lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.lt.adam$valid_xent_qwk), lwd=1.5, col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_xent_qwk),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names, lwd=1.5, col=preset.cols, bty="n", cex=0.8)
# valid exp qwk
plot(get.loess(afx.adam$valid_exp_qwk),type="l", lwd=1.5, col=preset.cols[1], xlab="epoch", ylab="valid QWK (exp)", ylim=c(0.8, 0.96))
lines(get.loess(afx.pois.lt.adam$valid_exp_qwk), lwd=1.5, col=preset.cols[2])
lines(get.loess(afemd2.pois.lt.adam$valid_exp_qwk), lwd=1.5, col=preset.cols[3])
lines(get.loess(afx.binom.lt.adam$valid_exp_qwk), lwd=1.5, col=preset.cols[4])
lines(get.loess(afemd2.binom.lt.adam$valid_exp_qwk), lwd=1.5, col=preset.cols[5])
lines(get.loess(afsqclassic.adam$valid_exp_qwk),lwd=1.5, col=preset.cols[6])
legend("bottomright", legend=exp.names, lwd=1.5, col=preset.cols, bty="n", cex=0.8)
dev.off()

