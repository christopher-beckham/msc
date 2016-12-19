

dfb2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.1.txt")[1:250,]
dfb2_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.2.txt")[1:250,]
dfb2_s3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop.3.txt")[1:250,]


dfb2_klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.1.txt")
dfb2_klo_s2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.2.txt")
#dfb2_klo_s3 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_klo_vxent.3.txt")


dfb2_qwk = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_qwk.1.txt")

# basic qwk from scratch

# should i resume the qwk experiment?
plot(dfb2$valid_kappa_exp,type="l",lwd=2)
lines(dfb2_klo$valid_kappa_exp,type="l",lwd=2,col="red")
lines(dfb2_qwk$valid_kappa_exp,lwd=2,col="orange")
legend("bottomright", legend=c("cross-entropy", "fix 'a'", "qwk"), col=c("black","red","orange"), lty="solid", lwd=2)

# show that the probability dist is bad
plot(dfb2$avg_valid_loss,type="l",lwd=2, ylim=c(0,10))
lines(dfb2_klo$avg_valid_loss,type="l",lwd=2,col="red")
lines(dfb2_qwk$avg_valid_loss,type="l",lwd=2,col="orange")

# this motivates the resume experiment. start off with x-ent and then make it quadratic kappa

dfb2_qwk_resume = dfb2[1:150,]
tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_with_qwk.1.txt",header=F)
colnames(tmp) = colnames(dfb2)
dfb2_qwk_resume = rbind(dfb2_qwk_resume, tmp)
dfb2_qwk_resume$valid_kappa_exp[1:150] = NA

dfb2_qwk_resume_s2 = dfb2[1:150,]
tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_with_qwk.2.txt",header=F)
colnames(tmp) = colnames(dfb2)
dfb2_qwk_resume_s2 = rbind(dfb2_qwk_resume_s2, tmp)
dfb2_qwk_resume_s2$valid_kappa_exp[1:150] = NA

dfb2_qwkcf_resume_s1 = dfb2[1:150,]
tmp = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/output_quadrant/low_res_n2_baseline_crop_resume_with_qwkcf_bs512.1.txt",header=F)
colnames(tmp) = colnames(dfb2)
dfb2_qwkcf_resume_s1 = rbind(dfb2_qwkcf_resume_s1, tmp)
dfb2_qwkcf_resume_s1$valid_kappa_exp[1:150] = NA

pdf("qwk_resume.pdf", width=8, height=5)

par(mfrow=c(1,2))

plot(dfb2$avg_valid_loss,type="l",lwd=2, ylim=c(0,8), xlim=c(150,250), xlab="epoch", ylab="valid loss")
lines(dfb2_s2$avg_valid_loss,col="black",lwd=2)
lines(dfb2_klo$avg_valid_loss,type="l",lwd=2,col="red")
lines(dfb2_klo_s2$avg_valid_loss,type="l",lwd=2,col="red")
lines(dfb2_qwk_resume$avg_valid_loss,lwd=2,col="blue")
lines(dfb2_qwk_resume_s2$avg_valid_loss,lwd=2,col="blue")
lines(dfb2_qwk$avg_valid_loss,lwd=2,col="purple")
legend("topright", 
       legend=c("cross-entropy", "fix 'a'", "qwk (warm start)", "qwk (cold start)"), 
       col=c("black","red","blue","purple"), lty="solid", lwd=2, cex=0.7)


plot(dfb2$valid_kappa_exp,type="l",lwd=2, ylim=c(0.5, 0.8), xlim=c(150,250), xlab="epoch", ylab="valid kappa")
lines(dfb2_s2$valid_kappa_exp,col="black",lwd=2)
lines(dfb2_klo$valid_kappa_exp,type="l",lwd=2,col="red")
lines(dfb2_klo_s2$valid_kappa_exp,type="l",lwd=2,col="red")
lines(dfb2_qwk_resume$valid_kappa_exp,lwd=2,col="blue")
lines(dfb2_qwk_resume_s2$valid_kappa_exp,lwd=2,col="blue")
lines(dfb2_qwk$valid_kappa_exp,lwd=2,col="purple")
legend("bottomright", 
       legend=c("cross-entropy", "fix 'a'", "qwk (warm start)", "qwk (cold start)"), 
       col=c("black","red","blue","purple"), lty="solid", lwd=2, cex=0.7)

dev.off()

# this motivates the resume experiment. start off with x-ent and then make it quadratic kappa

# ----------------

vd.baseline = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/valid_dist/low_res_n2_baseline_crop.1.txt",header=F)
vd.baseline.mat = as.matrix(vd.baseline[1:nrow(vd.baseline),1:5])
vd.baseline.probs = vd.baseline.mat[ cbind( 1:nrow(vd.baseline.mat), vd.baseline$V6) ]

vd.resume = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/valid_dist/low_res_n2_baseline_crop_resume_with_qwk.1.txt",header=F)
vd.resume.mat = as.matrix(vd.resume[1:nrow(vd.resume),1:5])
vd.resume.probs = vd.resume.mat[ cbind( 1:nrow(vd.resume.mat), vd.resume$V6) ]

vd.klo = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/diabetic_retinopathy/valid_dist/low_res_n2_baseline_crop_klo_vxent.2.txt",header=F)
vd.klo.mat = as.matrix(vd.klo[1:nrow(vd.klo),1:5])
vd.klo.probs = vd.klo.mat[ cbind( 1:nrow(vd.klo.mat), vd.klo$V6) ]

pdf("qwk_hist.pdf", width=8,height=3)
par(mfrow=c(1,3))
hist(vd.baseline.probs,breaks=20, ylim=c(0,300), main="cross-entropy", xlab="p")
hist(vd.klo.probs,breaks=20, ylim=c(0,300), main="fix 'a'", xlab="p")
hist(vd.resume.probs,breaks=20, ylim=c(0,300), main="qwk (warm start)", xlab="p")
dev.off()

# ---------

pdf("pdist_boxplots.pdf", width=8, height=3)
par(mfrow=c(1,3))
boxplot(vd.baseline[,1:5], names=c(0,1,2,3,4), ylab="p", main="cross-entropy")
boxplot(vd.klo[,1:5], names=c(0,1,2,3,4), ylab="p", main="fix 'a'")
boxplot(vd.resume[,1:5], names=c(0,1,2,3,4), ylab="p", main="qwk (warm start)")
dev.off()


