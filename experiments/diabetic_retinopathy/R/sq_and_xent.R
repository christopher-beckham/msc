dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:250,]

dfb2_qwkxe = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_learn-end-keep-xent.1.txt")

plot(dfb2$avg_valid_loss,type="l")
lines(dfb2_qwkxe$avg_valid_loss,col="red")

plot(dfb2$valid_kappa_exp,type="l")
lines(dfb2_qwkxe$valid_kappa_exp,col="red")

