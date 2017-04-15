tmp = read.csv("~/Desktop/cuda4_4/github/gan_stuff/pascal/output/vgg16_fixed_bn_gan_test2_bs16_fp/results.txt")


dr.qwk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_qwk_l2-1e-4_sgd_pre_split_hdf5//results.txt")
dr.qwk.2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/dr_qwk_l2-1e-4_sgd_pre_split_hdf5.2//results.txt")

ad.qwk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_qwk_l2-1e-4_sgd_pre_split_hdf5/results.txt")
ad.qwkreform.lr0.1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_qwkreform_l2-1e-4_sgd_pre_split_hdf5_lr0.1//results.txt")
ad.qwkreform.cls.lr0.1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/output/adience_qwkreform-classic_l2-1e-4_sgd_pre_split_hdf5_lr0.1//results.txt")

par(mfrow=c(2,1))

plot(ad.qwk$valid_xent_qwk,type="l", ylim=c(0,0.8), main="adience") # operates on p(y|x)
lines(ad.qwkreform.lr0.1$valid_xent_qwk,col="red") # operates on p(c|x) with learned 'a'
lines(ad.qwkreform.cls.lr0.1$valid_xent_qwk,col="blue") # operates on p(c|x) with learned 'a'
legend("bottomright", legend=c("ad.qwk","ad.qwkreform.lr0.1","ad.qwkreform.cls.lr0.1"),fill=c("black","red","blue"), cex=0.5,bty="n")

plot(ad.qwk$valid_exp_qwk,type="l", ylim=c(0,0.8), main="adience") # operates on p(y|x) with fixed 'a'
lines(ad.qwkreform.lr0.1$valid_exp_qwk,col="red") # operates on p(c|x) with learned 'a'
lines(ad.qwkreform.cls.lr0.1$valid_exp_qwk,col="blue") # operates on p(c|x) with learned 'a'
legend("bottomright", legend=c("ad.qwk","ad.qwkreform.lr0.1","ad.qwkreform.cls.lr0.1"),fill=c("black","red","blue"), cex=0.5,bty="n")

par(mfrow=c(1,1))

plot(ad.qwk$valid_loss,type="l")
lines(ad.qwkreform.lr0.1$valid_loss,col="red")
lines(ad.qwkreform.cls.lr0.1$valid_loss,col="blue")

## -----------------------------

par(mfrow=c(2,1))

plot(dr.qwk$valid_xent_qwk,type="l", ylim=c(0,0.8), main="dr") # operates on p(y|x)
lines(dr.qwk.2$valid_xent_qwk,col="red") # operates on p(c|x) with learned 'a'

plot(dr.qwk$valid_exp_qwk,type="l", ylim=c(0,0.8), main="dr") # operates on p(y|x) with fixed 'a'
lines(dr.qwk.2$valid_exp_qwk,col="red") # operates on p(c|x) with learned 'a'


