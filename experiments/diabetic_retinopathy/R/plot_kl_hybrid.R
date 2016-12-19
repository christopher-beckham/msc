dfb2_1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:200,]
dfb2_2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.2.txt")[1:200,]
dfb2_avg = (dfb2_1+dfb2_2)/2
# -----------
dfb2_kl0p1_1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl0.1.1.txt")[1:200,]
dfb2_kl0p1_2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl0.1.2.txt")[1:200,]
dfb2_kl0p1_avg = (dfb2_kl0p1_1+dfb2_kl0p1_2)/2
# -----------
dfb2_kl0p5_1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl0.5.1.txt")[1:200,]
dfb2_kl0p5_2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl0.5.2.txt")
dfb2_kl0p5_avg = dfb2_kl0p5_1 # todo: average
# -----------
dfb2_kl1p0_1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl1.0.1.txt")[1:200,]
dfb2_kl1p0_2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_kl1.0.2.txt")[1:200,]
dfb2_kl1p0_avg = (dfb2_kl1p0_1+dfb2_kl1p0_2) / 2
# ----------
dfb2_klo_1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")[1:200,]
dfb2_klo_2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.2.txt")[1:200,]
dfb2_klo_avg = (dfb2_klo_1+dfb2_klo_2) / 2

plot(dfb2_avg$avg_valid_loss, type="l", lwd=1.5, ylim=c(0.4,2.0), xlab="epoch", ylab="valid loss")
lines(dfb2_kl0p1_avg$avg_valid_loss, col="green", lwd=1.5)
lines(dfb2_kl0p5_avg$avg_valid_loss, col="red", lwd=1.5)
lines(dfb2_kl1p0_avg$avg_valid_loss, col="orange", lwd=1.5)
lines(dfb2_klo_avg$avg_valid_loss, col="purple", lwd=1.5)
legend(
  "topright",
  legend=c("x-ent","x-ent + 0.1*sq-err-exp", "x-ent + 0.5*sq-err-exp", "x-ent + 1.0*sq-err-exp", "sq-err-exp"),
  col=c("black", "green", "red", "orange", "purple"),
  lty=c("solid", "solid", "solid", "solid", "solid"),
  lwd=c(1.5,1.5,1.5,1.5),
  cex=0.8
)