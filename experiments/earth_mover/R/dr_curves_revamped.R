source("helpers.R")

# --------
# dr tau=1
# --------
# x-ent
dfx.adam =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")[1:150,]
dfx.adam.s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s2/dr_xent_l2-1e-4_sgd_pre_split_hdf5_adam//results.txt")[1:150,]
# x-ent + binom + tau1
dfx.binom.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam//results.txt")[1:150,]
# x-ent + pois + tau1
# (didn't decrease LR a second time like most other exps)
dfx.pois.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")[1:150,]
# emd2 + pois + tau1
dfemd2.pois.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# emd2 + binom + tau1
dfemd2.binom.t1.0.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt") ##TODO
dfemd2.binom.t1.0.adam.s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s2/dr_binom_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# sq-classic
dfsqclassic.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/dr_sqclassic-backrelu_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")

tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-1_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt.keras2") 

fig.width = 12
fig.height = 2.5

pdf("dr_tau1_camera-ready.pdf", width=fig.width, height=fig.height)
par(mar=c(5,4,1,1)+0.1, mfrow=c(1,4))
exp.names = c("x-ent (tau=1)",
              "x-ent + pois (tau=1)", 
              "emd2 + pois (tau=1)",
              "x-ent + binom (tau=1)",
              "emd2 + binom (tau=1)",
              "sq-err")
exp.cols = c(preset.cols[1], preset.cols[2], preset.cols[2], preset.cols[3], preset.cols[3], preset.cols[4])
exp.ltys = c("solid", "solid", "dashed", "solid", "dashed", "solid")
dr.tau1.dfs = list(dfx.adam, dfx.pois.t1.0.adam, dfemd2.pois.t1.0.adam, dfx.binom.t1.0.adam, dfemd2.binom.t1.0.adam, dfsqclassic.adam)
do.plot(dr.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_accuracy", "epoch", "valid acc (argmax)", c(0,150), c(0.6, 0.82))
do.plot(dr.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_accuracy", "epoch", "valid acc (exp)", c(0,150), c(0.6, 0.82))
do.plot(dr.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_qwk", "epoch", "valid QWK (argmax)", c(0,150), c(0.5, 0.75))
do.plot(dr.tau1.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_qwk", "epoch", "valid QWK (exp)", c(0,150), c(0.5, 0.75))
dev.off()


# --------------
# dr tau=learned
# --------------

# x-ent + pois + learn tau
dfx.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")[1:150,]
# x-ent + binom + learn tau
dfx.binom.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-learnsig_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

# final stretch
dfemd2.pois.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
# choked...need to wait a bit
dfemd2.binom.lt.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-learnsig_emd2_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt") ##TODO

pdf("dr_taulearned_camera-ready.pdf", width=fig.width, height=fig.height)
par(mar=c(5,4,1,1)+0.1, mfrow=c(1,4))
exp.names = c("x-ent (tau=1)",
              "x-ent + pois (tau=learned)", 
              "emd2 + pois (tau=learned)",
              "x-ent + binom (tau=learned)",
              "emd2 + binom (tau=learned)",
              "sq-err")
exp.cols = c(preset.cols[1], preset.cols[2], preset.cols[2], preset.cols[3], preset.cols[3], preset.cols[4])
exp.ltys = c("solid", "solid", "dashed", "solid", "dashed", "solid")
dr.lt.dfs = list(dfx.adam, dfx.pois.lt.adam, dfemd2.pois.lt.adam, dfx.binom.lt.adam, dfemd2.binom.lt.adam, dfsqclassic.adam)
do.plot(dr.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_accuracy", "epoch", "valid acc (argmax)", c(0,150), c(0.6, 0.82))
do.plot(dr.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_accuracy", "epoch", "valid acc (exp)", c(0,150), c(0.6, 0.82))
do.plot(dr.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_xent_qwk", "epoch", "valid QWK (argmax)", c(0,150), c(0.5, 0.75))
do.plot(dr.lt.dfs, exp.names, exp.cols, exp.ltys, "valid_exp_qwk", "epoch", "valid QWK (exp)", c(0,150), c(0.5, 0.75))
dev.off()

# ------


tmp7 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-learnsigfn-fixed_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
tmp8 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-fnlearn-simple_xent_l2-1e-4_sgd_pre_split_hdf5_adam//results.txt") # did div by mistake
tmp9 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-fnlearn-simple2_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt") # does mul

tmp10 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-learnsigfn-fixed-5-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam//results.txt")
tmp11 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_pois_t-learnsigfn-fixed-5l-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam//results.txt")
tmp12 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-learnsigfn-fixed-5-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

tmp13 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-fnlearn-simple-bd-5-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

dfx.binom.t1.0.5.adam = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_binom_t-1_5-1_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")

plot(dfx.adam$valid_exp_qwk,type="l"); 
#lines(tmp11$valid_exp_qwk,col="purple")
#lines(tmp10$valid_exp_qwk,col="brown")
#lines(tmp12$valid_exp_qwk,col="blue")
lines(dfx.binom.t1.0.adam$valid_exp_qwk,col="purple",lwd=1.5)
lines(dfx.binom.t1.0.5.adam$valid_exp_qwk,col="orange")
lines(dfx.pois.t1.0.adam$valid_exp_qwk,col="green")

plot(dfx.adam$valid_xent_qwk,type="l"); 
#lines(tmp11$valid_xent_qwk,col="purple")
#lines(tmp10$valid_xent_qwk,col="brown")
lines(tmp12$valid_xent_qwk,col="blue")
lines(dfx.binom.t1.0.adam$valid_xent_qwk,col="purple",lwd=1.5)
lines(dfx.binom.t1.0.5.adam$valid_xent_qwk,col="orange")

# --

plot(dfx.binom.t1.0.adam$valid_exp_qwk,type="l"); lines(tmp12$valid_exp_qwk,col="blue")
