#dfk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_klo_lr0.1.1.txt")

# --------------------
# compare x-ent vs klo
# --------------------

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:250,]
dfb2_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.2.txt")[1:250,]

dfb2_klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")
dfb2_klo_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.2.txt")

dfb2_lend_softmax = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end.1.txt")
dfb2_lend_softmax_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end.2.txt")

dfb2_hacky_ord = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.1.txt")
dfb2_hacky_ord_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_hacky_ordinal.2.txt")

dfb2_avg = (dfb2[1:250,]+dfb2_s2[1:250,])/2
dfb2_klo_avg=(dfb2_klo[1:250,]+dfb2_klo_s2[1:250,])/2
dfb2_lend_softmax_avg=(dfb2_lend_softmax[1:250,]+dfb2_lend_softmax_s2[1:250,])/2
dfb2_hacky_ord_avg=(dfb2_hacky_ord[1:250,]+dfb2_hacky_ord_s2[1:250,])/2


pdf("poster_nips.pdf", width=10, height=5)
par(mfrow=c(1,2))

plot(dfb2_avg$avg_valid_loss, type="l", col="black", lwd=1.5, ylim=c(0.4,3.0), xlab="epoch", ylab="valid loss")
lines(dfb2_klo_avg$avg_valid_loss, col="red", lwd=1.5)
legend(
  "topright",
  legend=c("cross-entropy","fix 'a'"),
  col=c("black", "red"),
  lty="solid",
  lwd=1.5
)

plot(dfb2_avg$valid_kappa_exp, type="l", col="black", lwd=1.5, xlim=c(150, 250), ylim=c(0.6,0.78), ylab="valid kappa", xlab="epoch")
lines(dfb2_klo_avg$valid_kappa_exp, col="red", lwd=1.5)
lines(dfb2_lend_softmax_avg$valid_kappa_exp, col="orange", lwd=1.5)
lines(dfb2_hacky_ord_avg$valid_kappa, col="green", lwd=1.5)
legend(
  "bottomright",
  legend=c("cross-entropy","fix 'a'","learn 'a'", "cheng"),
  col=c("black","red","orange","green"),
  lty=c("solid", "solid", "solid", "solid"),
  lwd=c(1.5,1.5,1.5,1.5)
)

dev.off()

# -----

# 3) The comments around learning ‘a’ not doing better could be because it’s overdetermined. 
# It would be interesting to see how a probability output unit scaled by the maximum ordinal value performs.

dfb2_lend_softmax_scaled = 
  read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s.1.txt")
dfb2_lend_softmax_scaled_s2 = 
  read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end_sigm-s.2.txt")


plot(dfb2$valid_kappa_exp,type="l")
lines(dfb2_klo$valid_kappa_exp,type="l",col="red")
lines(dfb2_lend_softmax$valid_kappa_exp,col="blue") # learn end with softmax
lines(dfb2_lend_softmax_scaled$valid_kappa_exp,col="purple",lwd=2)
lines(dfb2_lend_softmax_scaled_s2$valid_kappa_exp,col="purple",lwd=2)

