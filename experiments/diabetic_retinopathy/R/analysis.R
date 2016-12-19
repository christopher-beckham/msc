# zmuv appeared to help with convergence
df = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/resnet_only_augment_b64_zmuv.1.txt")
df2 = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/resnet_only_augment_b64_zmuv_l2.1.txt")

# we can see that at epoch=40 is when it starts to overfit
# maybe we can resume this experiment but set the learning
# rate to 0.001 and see what happens?
plot(df$avg_valid_loss, type="l")
abline(v=40, lty="dotted")

plot(df$valid_kappa, type="l")
abline(v=40, lty="dotted")
print(df$valid_kappa[40]) # 0.66 kappa

# -------------

plot(df$train_loss, type="l")
lines(df2$train_loss, col="red")

plot(df$avg_valid_loss, type="l")
lines(df2$avg_valid_loss, col="red")

plot(df$valid_kappa, type="l")
lines(df2$valid_kappa, col="red")

# fixes:
# - l2 bug
# - now y_left and y_right labels

df3 = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/new_only_augment_b64_zmuv_l2.1.txt")



#df4 = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/new_only_augment_b64_zmuv_l2_llr.1.txt")

# fixes:
# add data augmentation

df3a = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/new_only_augment_b64_zmuv_l2_more-aug.1.txt",header=FALSE)
tmp = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/new_only_augment_b64_zmuv_l2_more-aug_absorb.1.txt",header=FALSE)
df3a = rbind(df3a, tmp)
df3b = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/new_only_augment_b64_zmuv_l2_keras-aug.1.txt")
tmp = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/new_only_augment_b64_zmuv_l2_keras-aug_absorb_fsm_d0.5.1.txt",header=FALSE)
colnames(tmp) = colnames(df3b)
df3b = rbind(df3b, tmp)

plot(df3a$V3, type="l", lwd=1.5, xlim=c(0,250), ylim=c(1.0,2.0))
lines(df3b$avg_valid_loss, col="red", lwd=1.5)

# 256x2 resnet
df4 = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res.1.txt")
# 256x2 resnet with kappa coef = 0.01
df5 = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_kappa_hybrid_01.1.txt")
# 256x2 resnet with kappa coef = 0.001
df6 = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_kappa_hybrid_001.1.txt")
#plot(df4$train_loss, type="l", lwd=1.5)
plot(df4$avg_valid_loss, lwd=1.5, type="l", xlim=c(0,200))
#lines(df5$train_loss, col="red", lwd=1.5)
#lines(df5$avg_valid_loss, col="red", lwd=1.5)
#lines(df6$train_loss, col="orange", lwd=1.5)
#lines(df6$avg_valid_loss, col="orange", lwd=1.5)

plot(df4$valid_kappa_exp, type="l", lwd=1.5)
lines(df5$valid_kappa_exp, col="red", lwd=1.5)
lines(df6$valid_kappa_exp, col="orange", lwd=1.5)

#df4 = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res.1.txt")
#df4_redo = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_fsm.1.txt")
df4_redo_drop = read.csv("/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_d0.5_fsm.1.txt")
plot(df4_redo_drop$avg_valid_loss, lwd=1.5, type="l")
#lines(df4_redo_drop$avg_valid_loss, lwd=1.5, col="red")