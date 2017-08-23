df =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreform-classic-sm_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")

df2 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreform-classic_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")

df3=read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreform-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")
df3f.s1=read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreformf2-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")

df3.lp=read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreform-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_adam_lrp//results.txt") 
xent =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")
xent.lp =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_xent_l2_1e4_sgd_pre_split_hdf5_adam_lrp//results.txt")
sq = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_sqclassic-backrelu_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")
qwk =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_logqwk_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")
exp.s1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_exp_l2_1e4_sgd_pre_split_hdf5_adam//results.txt")

xent.50 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_xent_l2_1e4_sgd_pre_split_hdf5_50-50_adam/results.txt")
df3.50 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreform-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_50-50_adam/results.txt")
sq.50 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_sqclassic-backrelu_l2_1e4_sgd_pre_split_hdf5_50-50_adam/results.txt")
df3f.50.s1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreformf2-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_50-50_adam/results.txt")
exp.50.s1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_exp_l2_1e4_sgd_pre_split_hdf5_50-50_adam//results.txt")



tmp =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreformf2-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")

df3fp1.50 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreformfp1-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_50-50_adam/results.txt")


xent.25 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_xent_l2_1e4_sgd_pre_split_hdf5_25-75_adam/results.txt")
df3.25 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreform-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_25-75_adam/results.txt")
sq.25 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_sqclassic-backrelu_l2_1e4_sgd_pre_split_hdf5_25-75_adam/results.txt")
exp.25.s1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_exp_l2_1e4_sgd_pre_split_hdf5_25-75_adam/results.txt")

df3f.25.s1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreformf2-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_25-75_adam/results.txt")
df3f.25.s2 =read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s2/adience_qwkreformf-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_25-75_adam/results.txt")


xent.10 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_xent_l2_1e4_sgd_pre_split_hdf5_10-90_adam/results.txt")
df3.10 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_qwkreform-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_10-90_adam/results.txt")
sq.10 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/adience_sqclassic-backrelu_l2_1e4_sgd_pre_split_hdf5_10-90_adam/results.txt")


plot(df3$valid_exp_accuracy,type="l",ylim=c(0,1))
#lines(df2$valid_exp_accuracy,col="red")
#lines(df3$valid_exp_accuracy,col="red")

plot(df3$valid_exp_qwk,type="l",ylim=c(0,1))
#lines(df2$valid_exp_qwk,col="red")
#lines(df3$valid_exp_qwk,col="red")


###########

# full sized dataset

# our method performs the worst for accuracy
plot(xent$valid_xent_accuracy,type="l", main="Adience (90%-10%)")
lines(xent$valid_exp_accuracy,lty="dotted")
lines(df3$valid_exp_accuracy,col="red")
lines(sq$valid_exp_accuracy,col="blue")
lines(exp.s1$valid_exp_accuracy,col="purple")
lines(df3f.s1$valid_exp_accuracy,col="brown")
#lines(df3f.s1$valid_exp_accuracy,col="brown")

# marginally 'beats' x-ent, but it doesn't
# really look like a win
plot(xent$valid_xent_qwk,type="l")
lines(xent$valid_exp_qwk,lty="dotted")
lines(df3$valid_exp_qwk,col="red")
lines(sq$valid_exp_qwk,col="blue")
lines(exp.s1$valid_exp_qwk,col="purple")
lines(df3f.s1$valid_exp_qwk,col="brown")

plot(df3f.s1$valid_loss,type="l")

plot(sq$valid_loss,type="l")

# 50-50 dataset

# our method beats sq-err for accuracy
plot(df3.50$valid_exp_accuracy,type="l",col="red", ylim=c(0,1.), main="Adience (50%-50%)", ylab="valid accuracy", xlab="epoch")
lines(xent.50$valid_xent_accuracy,col="black")
lines(xent.50$valid_exp_accuracy,col="black",lty="dotted")
lines(sq.25$valid_exp_accuracy,col="blue")
lines(exp.50.s1$valid_exp_accuracy,col="purple")
lines(df3f.50.s1$valid_exp_accuracy,col="brown")
#lines(df3f.50.s2$valid_exp_accuracy,col="brown")

# our method beats sq-err/x-ent for qwk
plot(df3.50$valid_exp_qwk,type="l",ylim=c(0.7,0.95),col="red", main="Adience (50%-50%)", ylab="valid QWK", xlab="epoch")
lines(xent.50$valid_exp_qwk,col="black")
lines(xent.50$valid_xent_qwk,col="black", lty="dotted")
lines(sq.25$valid_exp_qwk,col="blue")
lines(exp.50.s1$valid_exp_qwk,col="purple")
lines(df3f.50.s1$valid_exp_qwk,col="brown")
#lines(df3f.50.s2$valid_exp_qwk,col="brown")

plot(xent.50$valid_loss,type="l",xlim=c(0,70))

plot(df3f.50.s1$valid_loss,type="l")

plot(sq.50$valid_loss,type="l")

plot(exp.50.s1$valid_loss,type="l")

# 25-75 data

# our method does worst for accuracy
plot(df3.25$valid_exp_accuracy,col="red",type="l",ylim=c(0,0.8),xlim=c(0,100),main="Adience (25%-75%)",ylab="valid accuracy", xlab="epoch")
lines(xent.25$valid_xent_accuracy,col="black")
lines(xent.25$valid_exp_accuracy,col="black",lty="dotted")
lines(sq.25$valid_exp_accuracy,col="blue")
#lines(df3fp1.25$valid_exp_accuracy,col="purple")
lines(df3f.25.s1$valid_exp_accuracy,col="brown")
#lines(df3f.25.s2$valid_exp_accuracy,col="brown")

# TODO: re-running this
plot(df3.25$valid_exp_qwk,type="l",ylim=c(0.4,0.95),col="red",xlim=c(0,100), main="Adience (25%-75%)",ylab="valid QWK", xlab="epoch")
lines(xent.25$valid_exp_qwk,col="black")
lines(xent.25$valid_xent_qwk,col="black", lty="dotted")
lines(sq.25$valid_exp_qwk,col="blue")
lines(df3f.25.s1$valid_exp_qwk,col="brown")
#lines(df3f.25.s2$valid_exp_qwk,col="brown")
#lines(df3fp1.25$valid_exp_qwk,col="purple")

plot(xent.25$valid_loss,type="l",xlim=c(0,100))

plot(df3f.25.s1$valid_loss,col="red",type="l")

plot(sq.25$valid_loss,type="l")


# -----------

xent.dr = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_xent_l2-1e-4_sgd_pre_split_hdf5_adam/results.txt")
qwk.dr.s1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/s1/dr_qwkreformf2-classic-smrelu_l2_1e4_sgd_pre_split_hdf5_adam/results.txt")

plot(xent.dr$valid_exp_accuracy,type="l")
lines(qwk.dr.s1$valid_exp_accuracy,col="brown")

plot(xent.dr$valid_exp_qwk,type="l")
lines(qwk.dr.s1$valid_exp_qwk,col="brown")

plot(qwk.dr.s1$valid_loss,type="l")

# ------------------

# 10-90 data?????

plot(df3.10$valid_exp_accuracy,col="red",type="l",ylim=c(0,1),main="Adience (10%-90%)",ylab="valid accuracy", xlab="epoch")
lines(xent.10$valid_xent_accuracy,col="black")
lines(sq.10$valid_xent_accuracy,col="blue")

plot(df3.10$valid_exp_qwk,col="red",type="l",ylim=c(0,1),main="Adience (10%-90%)",ylab="valid accuracy", xlab="epoch")
lines(xent.10$valid_xent_qwk,col="black")
lines(sq.10$valid_exp_qwk,col="blue")

plot(df3.10$valid_loss,type="l")

plot(xent.10$valid_loss,type="l")

plot(sq.10$valid_loss,type="l")

# ---

# if we assume y has been centered,
# so if y \in [1,2,3,4,5],
# y_centered \in [-1,-2,0,1,2]?

d = function(x,y=1) {
  print(y)
  num = (y*(x^2)) + (y^3) - (2*(x^2)*y)
  den = (x^4) + (2*(x^2)*(y^2)) + (y^4)
  return(num/den)
}


ground.truth=1
#xs = seq(from=0,to=ground.truth+5,by=0.1)
xs = seq(from=-1, to=2, by=0.1)
ys = d(x=xs,y=ground.truth)
plot(xs,ys,type="l",xlab="predicted x", ylab="dkappa/dx")
abline(v=ground.truth)
abline(h=0)


