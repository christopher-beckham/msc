#dfk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_klo_lr0.1.1.txt")

# --------------------
# compare x-ent vs klo
# --------------------

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:250,]
dfb2_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.2.txt")[1:250,]

rwk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_qwkcf_balanced_bs512.1.txt",header=FALSE)
colnames(rwk) = colnames(dfb2)
dfb2_resume_qwk = dfb2[1:150,]
dfb2_resume_qwk = rbind(dfb2_resume_qwk, rwk)

dfb2_klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")
dfb2_klo_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.2.txt")

dfb2_klo_resume = dfb2[1:25,]
tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_klo_at25epochs.1.txt",header=FALSE)[1:(200-25),]
colnames(tmp) = colnames(dfb2_klo_resume)
dfb2_klo_resume = rbind(
  dfb2_klo_resume,
  tmp
)
dfb2_klo_resume$avg_valid_loss[0:25] = rep(NA, 25)

dfb2_lend_softmax = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end.1.txt")
dfb2_lend_softmax_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end.2.txt")

dfb2_lend_softmax_scaled = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s.1.txt")
dfb2_lend_softmax_scaled_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s.2.txt")

dfb2_hacky_ord = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.1.txt")
dfb2_hacky_ord_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.2.txt")

dfb2_qwk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_qwk.1.txt")

pdf("p2_nips.pdf", width=8, height=5)
par(mfrow=c(1,2))

plot(dfb2$avg_valid_loss, type="l", col="black", lwd=1.5, ylim=c(0.4,3.0), xlab="epoch", ylab="valid loss")
lines(dfb2_s2$avg_valid_loss, col="black", lwd=1.5)
lines(dfb2_klo$avg_valid_loss, col="red", lwd=1.5)
lines(dfb2_klo_s2$avg_valid_loss, col="red", lwd=1.5)
lines(dfb2_lend_softmax_scaled$avg_valid_loss, col="orange", lwd=1.5)
lines(dfb2_qwk$avg_valid_loss, col="green", lwd=1.5)
legend(
  "topright",
  legend=c("cross-entropy","fix 'a'"),
  col=c("black", "red"),
  lty=c("solid", "solid"),
  lwd=c(1.5,1.5,1.5)
)

dfb2_avg = (dfb2[1:250,]+dfb2_s2[1:250,])/2
dfb2_klo_avg=(dfb2_klo[1:250,]+dfb2_klo_s2[1:250,])/2
dfb2_lend_softmax_avg=(dfb2_lend_softmax[1:250,]+dfb2_lend_softmax_s2[1:250,])/2
dfb2_hacky_ord_avg=(dfb2_hacky_ord[1:250,]+dfb2_hacky_ord_s2[1:250,])/2
dfb2_lend_softmax_scaled_avg=(dfb2_lend_softmax_scaled[1:250,]+dfb2_lend_softmax_scaled_s2[1:250,])/2

plot(dfb2_avg$valid_kappa_exp, type="l", col="black", lwd=1.5, xlim=c(150, 250), ylim=c(0.6,0.78), ylab="valid kappa", xlab="epoch")
lines(dfb2_klo_avg$valid_kappa_exp, col="red", lwd=1.5)
lines(dfb2_lend_softmax_avg$valid_kappa_exp, col="orange", lwd=1.5)
lines(dfb2_hacky_ord_avg$valid_kappa, col="green", lwd=1.5)
lines(dfb2_lend_softmax_scaled_avg$valid_kappa_exp, col="purple", lwd=1.5)
legend(
  "bottomright",
  legend=c("cross-entropy","fix 'a'","learn 'a'", "learn 'a' (sigm)", "cheng"),
  col=c("black","red","orange","purple","green"),
  lty=c("solid", "solid", "solid", "solid", "solid"),
  lwd=c(1.5,1.5,1.5,1.5),
  cex=0.8
)

#plot(dfb2$valid_kappa_exp, type="l", col="black", lwd=1.5, xlim=c(100, 250), ylim=c(0.5,0.75), ylab="valid kappa exp", xlab="epoch")
#lines(dfb2_s2$valid_kappa_exp, col="black", lwd=1.5)
#lines(dfb2_klo$valid_kappa_exp, col="red", lwd=1.5)
#lines(dfb2_klo_s2$valid_kappa_exp, col="red", lwd=1.5)
#lines(dfb2_lend_softmax$valid_kappa_exp, col="orange", lwd=1.5)
#lines(dfb2_lend_softmax_s2$valid_kappa_exp, col="orange", lwd=1.5)
##lines(dfb2_lend_sigm$valid_kappa_exp, col="purple", lwd=1.5)
##lines(dfb2_lend_relu$valid_kappa_exp, col="pink", lwd=1.5)
#lines(dfb2_hacky_ord$valid_kappa, col="green", lwd=1.5)
#lines(dfb2_hacky_ord_s2$valid_kappa, col="green", lwd=1.5)
#legend(
#  "bottomright",
#  legend=c("cross-entropy","fix 'a'","learn 'a'", "cheng"),
#  col=c("black","red","orange","green"),
#  lty=c("solid", "solid", "solid", "solid"),
#  lwd=c(1.5,1.5,1.5,1.5)
#)

#print(max(dfb2$valid_kappa_exp))
#print(max(dfb2_klo$valid_kappa_exp))
#print(max(dfb2_lend_softmax$valid_kappa_exp))
#print(max(dfb2_hacky_ord$valid_kappa))
#print(max(dfb2_klo_resume$valid_kappa_exp))

(max(dfb2$valid_kappa_exp) + max(dfb2_s2$valid_kappa_exp)) / 2
(max(dfb2_klo$valid_kappa_exp) + max(dfb2_klo_s2$valid_kappa_exp)) / 2
(max(dfb2_lend_softmax$valid_kappa_exp) + max(dfb2_lend_softmax_s2$valid_kappa_exp)) / 2
(max(dfb2_hacky_ord$valid_kappa) + max(dfb2_hacky_ord_s2$valid_kappa)) / 2

dev.off()

# test out two seed replicates

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:200,]
dfb2_klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")
dfb2_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.2.txt")[1:200,]
dfb2_klo_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.2.txt")

plot(dfb2$avg_valid_loss, type="l")
lines(dfb2_s2$avg_valid_loss, type="l")
lines(dfb2_klo$avg_valid_loss, type="l", col="red")
lines(dfb2_klo_s2$avg_valid_loss, type="l", col="red")

plot(dfb2$valid_kappa_exp, type="l")
lines(dfb2_s2$valid_kappa_exp, type="l")
lines(dfb2_klo$valid_kappa_exp, type="l", col="red")
lines(dfb2_klo_s2$valid_kappa_exp, type="l", col="red")


# ---------------------
# testing ord structure
# ---------------------

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:200,]
dfb2_ordv2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_ordinalv2.1.txt")
dfb2_ordv3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_ordinalv3.1.txt")
dfb2_ordv4 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_ordinalv4.1.txt")

plot(dfb2$avg_valid_loss, type="l", lwd=1.5)
lines(dfb2_ordv2$avg_valid_loss, col="red", lwd=1.5)
lines(dfb2_ordv3$avg_valid_loss, col="blue", lwd=1.5)
lines(dfb2_ordv4$avg_valid_loss, col="purple", lwd=1.5)

plot(dfb2$valid_kappa_exp, type="l", lwd=1.5)
lines(dfb2_ordv2$valid_kappa_exp, col="red", lwd=1.5)
lines(dfb2_ordv3$valid_kappa_exp, col="blue", lwd=1.5)
lines(dfb2_ordv4$avg_valid_loss, col="purple", lwd=1.5)

# ------------------
# testing kappa loss
# ------------------

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline.1.txt")
dfb2_kl0p1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl0.1.1.txt")
dfb2_kl0p5 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl0.5.1.txt")
dfb2_kl1p0 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl1.0.1.txt")
dfb2_klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")

pdf("p2_2.pdf")
par(mfrow=c(2,1))

plot(dfb2$avg_valid_loss, type="l", lwd=1.5, ylim=c(0.4,2.0), xlab="epoch", ylab="valid loss")
lines(dfb2_kl0p1$avg_valid_loss, col="green", lwd=1.5)
lines(dfb2_kl0p5$avg_valid_loss, col="red", lwd=1.5)
lines(dfb2_kl1p0$avg_valid_loss, col="orange", lwd=1.5)
lines(dfb2_klo$avg_valid_loss, col="purple", lwd=1.5)
legend(
  "topright",
  legend=c("x-ent","x-ent + 0.1*sq-err-exp", "x-ent + 0.5*sq-err-exp", "x-ent + 1.0*sq-err-exp", "sq-err-exp"),
  col=c("black", "green", "red", "orange", "purple"),
  lty=c("solid", "solid", "solid", "solid", "solid"),
  lwd=c(1.5,1.5,1.5,1.5),
  cex=0.8
)

plot(dfb2$valid_kappa_exp, type="l", lwd=1.5, xlab="epoch", ylab="valid loss")
lines(dfb2_kl0p1$valid_kappa_exp, col="green", lwd=1.5)
lines(dfb2_kl0p5$valid_kappa_exp, col="red", lwd=1.5)
lines(dfb2_kl1p0$valid_kappa_exp, col="orange", lwd=1.5)
lines(dfb2_klo$valid_kappa_exp, col="purple", lwd=1.5)
legend(
  "bottomright",
  legend=c("x-ent","x-ent + 0.1*sq-err-exp", "x-ent + 0.5*sq-err-exp", "x-ent + 1.0*sq-err-exp", "sq-err-exp"),
  col=c("black", "green", "red", "orange", "purple"),
  lty=c("solid", "solid", "solid", "solid", "solid"),
  lwd=c(1.5,1.5,1.5,1.5),
  cex=0.8
)

dev.off()



# ----------


dfb2_crop = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:200,]
dfb2_kl01 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl0.1.1.txt")
dfb2_klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")

plot(dfb2_crop$avg_valid_loss, type="l", lwd=1.5, ylim=c(0.4,2.0), xlab="epoch", ylab="valid loss")
lines(dfb2_kl01$avg_valid_loss, col="green", lwd=1.5)
lines(dfb2_klo$avg_valid_loss, col="purple", lwd=1.5)
legend(
  "topright",
  legend=c("x-ent","x-ent + 0.1*sq-err-exp", "sq-err-exp"),
  col=c("black", "green", "purple"),
  lty=c("solid", "solid", "solid"),
  lwd=c(1.5,1.5,1.5),
  cex=0.8
)



# ----------------------
# testing ord x-ent loss
# ----------------------

dfb2_224 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:200,]
dfb2_224_l1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_ordinalxent-1-redo-1w.1.txt")
#dfb2_kl2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_kl4.1.txt")

plot(dfb2_224$avg_valid_loss, type="l", lwd=1.5)
lines(dfb2_224_l1$avg_valid_loss, col="red", lwd=1.5)
#lines(dfb2_kl2$avg_valid_loss, col="orange", lwd=1.5)

plot(dfb2_224$valid_kappa_exp, type="l")
lines(dfb2_224_l1$valid_kappa_exp, col="red")
lines(dfb2_224_l2$valid_kappa_exp, col="orange")
lines(dfb2_224_l3$valid_kappa_exp, col="blue")

# ----------------------------
# testing baseline n=2,n=6,...
# ----------------------------

# - why is it that b6 > b4 > b2 in terms of val exp kappa
#   yet it is the opposite trend for val/train loss??
# - cropped versions seem to exhibit smaller val loss
#   but seem to do slightly worse on valid exp kappa

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline.1.txt")
dfb2_crop = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")
dfb4 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n4_baseline.1.txt")
dfb4_crop = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n4_baseline_crop.1.txt")
dfb6 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n6_baseline.1.txt")
dfb6_crop = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n6_baseline_crop.1.txt")

plot(dfb2$train_loss, type="l", lwd=1.5)
lines(dfb2_crop$train_loss, type="l", lwd=1.5, lty="dotted")
lines(dfb4$train_loss, col="red", lwd=1.5)
lines(dfb4_crop$train_loss, col="red", lwd=1.5, lty="dotted")
lines(dfb6$train_loss, col="orange", lwd=1.5)
lines(dfb6_crop$train_loss, col="orange", lwd=1.5, lty="dotted")

plot(dfb2$avg_valid_loss, type="l", lwd=1.5)
lines(dfb2_crop$avg_valid_loss, type="l", lwd=1.5, lty="dotted")
lines(dfb4$avg_valid_loss, col="red", lwd=1.5)
lines(dfb4_crop$avg_valid_loss, col="red", lwd=1.5, lty="dotted")
lines(dfb6$avg_valid_loss, col="orange", lwd=1.5)
lines(dfb6_crop$avg_valid_loss, col="orange", lwd=1.5, lty="dotted")

plot(dfb2$valid_kappa_exp, type="l", lwd=1.5)
lines(dfb4$valid_kappa_exp, col="red", lwd=1.5)
lines(dfb6$valid_kappa_exp, col="orange", lwd=1.5)


# --------------------------------------------------------------
# baselines that take individual left/right eye images (256/224)
# --------------------------------------------------------------

# n=2

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline.1.txt")
dfb2_224 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")
dfb2_ord = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_ordinal.1.txt")

plot(dfb2$train_loss, type="l", xlab="epoch", ylab="train loss")
lines(dfb2_224$train_loss, col="red")
lines(dfb2_ord$train_loss, col="orange")

plot(dfb2$avg_valid_loss, type="l", xlab="epoch", ylab="valid loss")
lines(dfb2_224$avg_valid_loss, col="red")
lines(dfb2_ord$avg_valid_loss, col="orange")

plot(dfb2$valid_kappa_exp, type="l", xlab="epoch", ylab="valid kappa")
lines(dfb2_224$valid_kappa_exp, col="red")
lines(dfb2_ord$valid_kappa, col="orange")

# n=4

dfb4 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n4_baseline.1.txt")
dfb4_224 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n4_baseline_crop.1.txt")

plot(dfb4$train_loss, type="l", xlab="epoch", ylab="train loss")
lines(dfb4_224$train_loss, col="red")

plot(dfb4$avg_valid_loss, type="l", xlab="epoch", ylab="valid loss")
lines(dfb4_224$avg_valid_loss, col="red")

# n=6

dfb6 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n6_baseline.1.txt")
dfb6_224 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n6_baseline_crop.1.txt")

plot(dfb6$train_loss, type="l", xlab="epoch", ylab="train loss")
lines(dfb6_224$train_loss, col="red")

plot(dfb6$avg_valid_loss, type="l", xlab="epoch", ylab="valid loss")
lines(dfb6_224$avg_valid_loss, col="red")

# --------------------------------------------------------
# baselines that take both eyes at the same time (256/224)
# --------------------------------------------------------
