# figure 1

pdf("exp1.pdf", width=8, height=6)
par(mfrow=c(1,2))

df = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/resnet-beefy_absorb_fsm_d0.5.1.txt")
tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/resnet-beefy_absorb_fsm_d0.5_dfsm.1.txt",header=FALSE)
#tmp = tmp[2:nrow(tmp),]
colnames(tmp) = colnames(df)
df = rbind(df,tmp)

dftest = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/high_res_dfsm_n2.1.txt")

plot(df$train_loss, type="l", lwd=1.5, xlab="epoch", ylab="train/valid loss")
lines(df$avg_valid_loss, col="red", lwd=1.5)
legend("topright", legend=c("train loss", "valid loss"), fill=c("black", "red"))
lines(dftest$avg_valid_loss, col="purple", lwd=1.5)

plot(df$valid_kappa, type="l", lwd=1.5, xlab="epoch", ylab="valid kappa")
lines(df$valid_kappa_exp, type="l", lwd=1.5, col="red")
lines(dftest$valid_kappa_exp, type="l", lwd=1.5, col="purple")
legend("bottomright", legend=c("valid kappa", "exp valid kappa"), fill=c("black", "red"))

dev.off()

# --------------------

pdf("exp_n46.pdf", width=8, height=6)
par(mfrow=c(1,2))

df4 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_dfsm_n4.1.txt")[1:200,]
df4b = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_beefy_dfsm_n4.1.txt")
df6 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_dfsm_n6.1.txt")[1:200,]
df6b = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_beefy_dfsm_n6.1.txt")

plot(df4$avg_valid_loss, type="l", lwd=1.5, col="blue", xlab="epoch", ylab="valid loss")
lines(df4b$avg_valid_loss, col="blue", lty="dashed", lwd=1.5)
lines(df6$avg_valid_loss, col="red", lwd=1.5)
lines(df6b$avg_valid_loss, col="red", lty="dashed", lwd=1.5)
legend("topright", legend=c("N=4,hf", "N=4", "N=6,hf", "N=6"), col=c("blue", "blue", "red", "red"),
       lty=c("solid", "dashed", "solid", "dashed"), lwd=c(1.5,1.5,1.5,1.5))

plot(df4$valid_kappa, type="l", lwd=1.5, col="blue", xlab="epoch", ylab="valid kappa")
lines(df4b$valid_kappa_exp, col="blue", lty="dashed", lwd=1.5)
lines(df6$valid_kappa, col="red", lwd=1.5)
lines(df6b$valid_kappa_exp, col="red", lty="dashed", lwd=1.5)
legend("bottomright", legend=c("N=4,hf", "N=4", "N=6,hf", "N=6"), col=c("blue", "blue", "red", "red"),
       lty=c("solid", "dashed", "solid", "dashed"), lwd=c(1.5,1.5,1.5,1.5))

dev.off()


# -----------------

pdf("exp_n6.pdf", width=8, height=6)
par(mfrow=c(1,2))

df6 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_fsm_n6.1.txt")[1:200,]
plot(df6$train_loss, type="l", lwd=1.5, xlab="epoch", ylab="train/valid loss")
lines(df6$avg_valid_loss, col="red", lwd=1.5)
legend("topright", legend=c("train loss", "valid loss"), fill=c("black", "red"))

plot(df6$valid_kappa, type="l", lwd=1.5, xlab="epoch", ylab="train/valid loss")
lines(df6$valid_kappa_exp, type="l", lwd=1.5, col="red")
legend("bottomright", legend=c("valid kappa", "exp valid kappa"), fill=c("black", "red"))

dev.off()

# ----------------

dfb4 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n4_baseline.1.txt")[1:200,]
dfb6 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n6_baseline.1.txt")[1:200,]

# - currently running dfb2
# - want to run best+kappa penalty

pdf("exp_baseline.pdf", width=8, height=6)
par(mfrow=c(1,2))

plot(dfb4$avg_valid_loss, type="l", lwd=1.5, xlab="epoch", ylab="valid loss", col="blue")
lines(dfb6$avg_valid_loss, col="red", lwd=1.5)
legend("topright", legend=c("n=4", "n=6"), fill=c("blue", "red"))

plot(dfb4$valid_kappa_exp, type="l", lwd=1.5, xlab="epoch", ylab="exp valid kappa", col="blue")
lines(dfb6$valid_kappa_exp, type="l", lwd=1.5, col="red")
legend("bottomright", legend=c("n=4", "n=6"), fill=c("blue", "red"))

dev.off()