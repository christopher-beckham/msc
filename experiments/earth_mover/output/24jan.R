df = read.csv("adience_baseline_pre_split/r.bak")

plot(df$valid_xent_accuracy,type="l")
lines(df$valid_exp_accuracy,col="red")

plot(df$valid_xent_qwk,type="l")

plot(df$valid_xent,type="l")

# --------

df2 = read.csv("adience_emd2_pre_split/results.txt")
df3 = read.csv("adience_xemd2_1e-1_pre_split/results.txt")

df4 = read.csv("adience_exp_pre_split/results.txt")

plot(df2$train_loss,type="l")

plot(df$valid_xent,type="l",ylim=c(0,4),lwd=2,xlim=c(0,40))
lines(df2$valid_xent,col="red",lwd=2)
lines(df3$valid_xent,col="green",lwd=2)

plot(df$valid_xent_qwk,type="l",ylim=c(0,1),lwd=2,xlim=c(0,40))
lines(df2$valid_xent_qwk,col="red",lwd=2)
lines(df3$valid_xent_qwk,col="green",lwd=2)

# --------------------------------------------------------------------------

get.loess = function(x, span=0.2) {
  y = 1:length(x)
  lo = loess(x~y,span=span)
  return(predict(lo))
}

dfx = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2-1e-4_adam_pre_split/results.txt")
dfemd2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_emd2_l2-1e-4_adam_pre_split/results.txt")
dfexp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_exp_l2-1e-4_adam_pre_split/results.txt")
dfemd22 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_emd22_l2-1e-4_adam_pre_split/results.txt")
dfqwk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_qwk_l2-1e-4_adam_pre_split/results.txt")

plot(dfx$valid_xent,type="l")
lines(dfemd2$valid_xent,col="red")
lines(dfemd22$valid_xent,col="purple")
lines(dfexp$valid_xent,col="blue")
lines(dfqwk$valid_xent,col="brown")

plot(dfx$valid_xent_qwk,type="l")
lines(dfemd2$valid_xent_qwk,col="red")
lines(dfemd22$valid_xent_qwk,col="purple")
lines(dfexp$valid_exp_qwk,col="blue")
lines(dfqwk$valid_exp_qwk,col="brown")

ad_xent_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2-1e-4_sgd_pre_split_hdf5/results.txt")
ad_emd2_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_emd2_l2-1e-4_sgd_pre_split_hdf5/results.txt")
ad_exp_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_exp_l2-1e-4_sgd_pre_split_hdf5/results.txt")


####

plot(get.loess(dfx$valid_xent),type="l")
lines(get.loess(dfemd2$valid_xent),col="red")
lines(get.loess(dfemd22$valid_xent),col="purple")
lines(get.loess(dfexp$valid_xent),col="blue")
lines(get.loess(dfqwk$valid_xent),col="brown")

plot(get.loess(dfx$valid_xent_qwk),type="l")
lines(get.loess(dfemd2$valid_xent_qwk),col="red")
lines(get.loess(dfemd22$valid_xent_qwk),col="purple")
lines(get.loess(dfexp$valid_exp_qwk),col="blue")
lines(get.loess(dfqwk$valid_exp_qwk),col="brown")

# ----


drxent_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_adam_pre_split_hdf5/results.txt")
drexp_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_exp_l2-1e-4_adam_pre_split_hdf5/results.txt")
dremd2_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_emd2_l2-1e-4_adam_pre_split_hdf5/results.txt")
drsoft_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_soft_l2-1e-4_adam_pre_split_hdf5/results.txt")


plot(drxent$train_loss,type="l")
lines(drxent_h5$train_loss,col="red")

plot(drxent_h5$valid_xent,type="l",xlim=c(0,200))
lines(drexp_h5$valid_xent,col="red")
lines(dremd2_h5$valid_xent,col="blue")
lines(drsoft_h5$valid_xent,col="brown")

plot(drxent$valid_exp_qwk,type="l",ylim=c(0,0.8),xlim=c(0,200))
lines(drxent_h5$valid_exp_qwk,col="red")
lines(dremd2_h5$valid_exp_qwk,col="blue")
lines(drsoft_h5$valid_exp_qwk,col="brown")

# ---

drxent_sgd_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_sgd_pre_split_hdf5//results.txt")
# todo: fix numbering
drexp_sgd_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_exp_l2-1e-4_sgd_pre_split_hdf5//results.txt")
dremd2_sgd_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_emd2_l2-1e-4_sgd_pre_split_hdf5//results.txt")
drsoft_sgd_h5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_soft_l2-1e-4_sgd_pre_split_hdf5//results.txt")

plot(drxent_sgd_h5$valid_xent,type="l")
lines(drexp_sgd_h5$valid_xent,col="red")
lines(dremd2_sgd_h5$valid_xent,col="blue")

plot(drxent_sgd_h5$valid_exp_qwk,type="l")
lines(drexp_sgd_h5$valid_exp_qwk,col="red")
lines(dremd2_sgd_h5$valid_exp_qwk,col="blue")
