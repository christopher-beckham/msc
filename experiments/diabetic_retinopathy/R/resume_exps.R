
dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:250,]

par(mfrow=c(1,1))

# qwk
dfb2_resume_qwk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:150,]
tmp1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_with_qwk.1.txt",header=F)
colnames(tmp1) = colnames(dfb2_resume_qwk)
dfb2_resume_qwk = rbind(dfb2_resume_qwk, tmp1)
plot(dfb2_resume_qwk$avg_valid_loss,type="l",col="red", xlim=c(100,200))
#plot(dfb2_resume_qwk$valid_kappa,type="l",col="red", xlim=c(100,200))

# qwk 512
dfb2_resume_qwk_512 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:150,]
tmp1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_with_qwk_bs512.1.txt",header=F)
colnames(tmp1) = colnames(dfb2_resume_qwk_512)
dfb2_resume_qwk_512 = rbind(dfb2_resume_qwk_512, tmp1)
lines(dfb2_resume_qwk_512$avg_valid_loss,type="l",col="blue")
#lines(dfb2_resume_qwk_512$valid_kappa,type="l",col="blue")

# qwk balanced 512
dfb2_resume_qwk_ba512 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:150,]
tmp1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_with_qwk_balanced_bs512.1.txt",header=F)
colnames(tmp1) = colnames(dfb2_resume_qwk_ba512)
dfb2_resume_qwk_ba512 = rbind(dfb2_resume_qwk_ba512, tmp1)
lines(dfb2_resume_qwk_ba512$avg_valid_loss,type="l", col="green")
#lines(dfb2_resume_qwk_ba512$valid_kappa,type="l", col="green")

# qwkcf 512 (implies balanced mb's)
dfb2_resume_qwkcf_512 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:150,]
tmp1 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_with_qwkcf_bs512.1.txt",header=F)
colnames(tmp1) = colnames(dfb2_resume_qwkcf_512)
dfb2_resume_qwkcf_512 = rbind(dfb2_resume_qwkcf_512, tmp1)
lines(dfb2_resume_qwkcf_512$avg_valid_loss,type="l",col="purple")
#lines(dfb2_resume_qwkcf_512$valid_kappa,type="l",col="purple")

#lines(dfb2$valid_kappa,lwd=2,col="black")

legend("topright", legend=c("qwk", "qwk512", "qwkb512", "qwkb512cf"), fill=c("red", "blue", "green", "purple"))

# --------

extract_prob_dists = function(df) {
  dist_cleaned = c()
  dist_idxs = df$V6
  df$V6 = NULL
  for(i in 1:nrow(df)) {
    dist_cleaned = c(dist_cleaned, df[i, dist_idxs[i]+1])
  }
  return(dist_cleaned)
}

par(mfrow=c(2,2))

dfb2_dist = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/valid_dist//low_res_n2_baseline_crop.1.txt",header=F)
dfb2_dist_cleaned = extract_prob_dists(dfb2_dist)
hist(dfb2_dist_cleaned,breaks=20, ylim=c(0,1000), main="x-ent")

# this experiment likes to put a lot of confidence in its predictions
dfb2_resume_dist = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/valid_dist//low_res_n2_baseline_crop_resume_with_qwkcf_bs512.1.txt",header=F)
dfb2_resume_dist_cleaned = extract_prob_dists(dfb2_resume_dist)
hist(dfb2_resume_dist_cleaned,breaks=20, ylim=c(0,1000), main="qwkcf bs512")


dfb2_resume2_dist = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/valid_dist//low_res_n2_baseline_crop_resume_with_qwk_balanced_bs512.1.txt",header=F)
dfb2_resume2_dist_cleaned = extract_prob_dists(dfb2_resume2_dist)
hist(dfb2_resume2_dist_cleaned,breaks=20, ylim=c(0,1000), main="qwk bs512")

dfb2_resume3_dist = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/valid_dist//low_res_n2_baseline_crop_resume_with_qwk.1.txt",header=F)
dfb2_resume3_dist_cleaned = extract_prob_dists(dfb2_resume3_dist)
hist(dfb2_resume3_dist_cleaned,breaks=20, ylim=c(0,1000), main="qwk")

#

boxplot(dfb2_dist$V1, dfb2_dist$V2, dfb2_dist$V3, dfb2_dist$V4, dfb2_dist$V5)
boxplot(dfb2_resume_dist$V1, dfb2_resume_dist$V2, dfb2_resume_dist$V3, dfb2_resume_dist$V4, dfb2_resume_dist$V5) #qwkcf 512
boxplot(dfb2_resume2_dist$V1, dfb2_resume2_dist$V2, dfb2_resume2_dist$V3, dfb2_resume2_dist$V4, dfb2_resume2_dist$V5) # qwkb 512
boxplot(dfb2_resume3_dist$V1, dfb2_resume3_dist$V2, dfb2_resume3_dist$V3, dfb2_resume3_dist$V4, dfb2_resume3_dist$V5) # qwk

# ---------

par(mfrow=c(1,1))

plot(dfb2$train_loss,type="l")
tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_qwk_hybrid1.1.txt")
tmp2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_qwk_hybrid0.1.1.txt")
tmp3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_qwk_hybrid0.01.1.txt")
plot(dfb2$valid_kappa_exp, type="l")
lines(tmp$valid_kappa_exp,type="l",col="red")
lines(tmp2$valid_kappa_exp,col="blue")
lines(tmp3$valid_kappa_exp,col="green")
lines(tmp2$valid_kappa_exp,col="blue")

# ----------

par(mfrow=c(1,2))

df_mx = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_mnist//xent1.txt")
df_mk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_mnist//kappa1.txt"

plot(df_mx$train_xent_loss,type="l", xlab="epoch", ylab="x-ent loss", lwd=2)
lines(df_mx$valid_xent_loss,type="l", lty="dotted", lwd=2)

plot(df_mx$train_kappa_loss, type="l", col="red", ylab="kappa loss", lwd=2, xlab="epoch")
lines(df_mx$valid_kappa_loss,type="l", lty="dotted", lwd=2, col="red")

# ---

plot(df_mk$train_xent_loss,type="l", xlab="epoch", ylab="x-ent loss", lwd=2)
lines(df_mk$valid_xent_loss,type="l", lty="dotted", lwd=2)

plot(df_mk$train_kappa_loss, type="l", col="red", ylab="kappa loss", lwd=2, xlab="epoch")
lines(df_mk$valid_kappa_loss,type="l", lty="dotted", lwd=2, col="red")



