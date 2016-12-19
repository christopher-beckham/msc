#dfk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_klo_lr0.1.1.txt")

# --------------------
# compare x-ent vs klo
# --------------------

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:200,]
dfb2_klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")

dfb2_klo_resume = dfb2[1:25,]
tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_klo_at25epochs.1.txt",header=FALSE)
colnames(tmp) = colnames(dfb2_klo_resume)
dfb2_klo_resume = rbind(
  dfb2_klo_resume,
  tmp
)

dfb2_lend_softmax = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end.1.txt")
dfb2_lend_softmax_scaled = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s.1.txt")

dfb2_lend_sigm = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-out.1.txt")
dfb2_lend_sigm_scaled = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s_sigm-out.1.txt")

dfb2_lend_relu_scaled = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s_relu-out.1.txt")
dfb2_lend_relu = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_relu-out.1.txt")

dfb2_hacky_ord = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.1.txt")

pdf("p2.pdf", height=8)
par(mfrow=c(3,1))

plot(dfb2$avg_valid_loss, type="l", lwd=1.5, ylim=c(0.4,3.0), xlab="epoch", ylab="valid loss")
lines(dfb2_klo$avg_valid_loss, col="red", lwd=1.5)
lines(dfb2_lend_softmax$avg_valid_loss, col="orange", lwd=1.5)
lines(dfb2_klo_resume$avg_valid_loss, col="brown", lwd=1.5)
abline(v=25, col="brown", lty="dotted")
# makes no sense to use sigmoid outs for this plot
#lines(dfb2_lend_sigm$avg_valid_loss, col="purple", lwd=1.5)
#lines(dfb2_lend_sigm_scaled$avg_valid_loss, col="purple", lwd=1.5, lty="dashed")
legend(
  "topright",
  legend=c("x-ent","sq-err-exp","resume"),
  col=c("black", "red", "brown"),
  lty=c("solid", "solid", "solid"),
  lwd=c(1.5,1.5,1.5)
)

plot(dfb2$valid_kappa_exp, type="l", col="black", lwd=1.5, ylim=c(0.3,0.75), xlab="epoch", ylab="valid kappa / valid kappa exp trick")
lines(dfb2$valid_kappa, type="l", col="black", lty="dashed", lwd=1.5)
lines(dfb2_klo$valid_kappa_exp, col="red", lwd=1.5)
lines(dfb2_klo$valid_kappa, col="red", lwd=1.5, lty="dashed")
legend(
  "bottomright",
  legend=c("x-ent w/ argmax","x-ent w/ exp trick","sq-err-exp w/ argmax", "sq-err-exp w/ exp trick"),
  col=c("black", "black", "red", "red"),
  lty=c("solid", "dashed", "solid", "dashed"),
  lwd=c(1.5,1.5,1.5,1.5)
)

plot(dfb2$valid_kappa_exp, type="l", col="black", lwd=2.0, ylim=c(0.1,0.75), ylab="valid kappa exp", xlab="epoch")
lines(dfb2_klo$valid_kappa_exp, col="red", lwd=2.0)
lines(dfb2_lend_softmax$valid_kappa_exp, col="orange", lwd=1.5)
#lines(dfb2_lend_sigm$valid_kappa_exp, col="purple", lwd=1.5)
#lines(dfb2_lend_relu$valid_kappa_exp, col="pink", lwd=1.5)
lines(dfb2_hacky_ord$valid_kappa, col="green", lwd=1.5)
lines(dfb2_klo_resume$valid_kappa_exp, col="brown", lwd=1.5)
abline(v=25, col="brown", lty="dotted")
#legend(
#  "bottomright",
#  legend=c("x-ent","sq-err-exp","learn-softmax", "learn-sigm", "learn-relu", "ord-enc", "resume"),
#  col=c("black","red","orange","purple","pink","green","brown"),
#  lty=c("solid", "solid", "solid", "solid", "solid", "solid", "solid"),
#  lwd=c(2.5,2.5,1.5,1.5,1.5,1.5)
#)
legend(
  "bottomright",
  legend=c("x-ent","sq-err-exp","learn-softmax", "ord-enc", "resume"),
  col=c("black","red","orange","green","brown"),
  lty=c("solid", "solid", "solid", "solid", "solid"),
  lwd=c(2.5,2.5,1.5,1.5)
)

print(max(dfb2$valid_kappa_exp))
print(max(dfb2_klo$valid_kappa_exp))
print(max(dfb2_lend_softmax$valid_kappa_exp))
print(max(dfb2_hacky_ord$valid_kappa))
print(max(dfb2_klo_resume$valid_kappa_exp))

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
