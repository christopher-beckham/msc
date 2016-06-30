pdf(file="relu_exp_1.pdf")

par(mfrow=c(3,1))

plot_lr_lines = function() {
  abline(v=which(df$epoch == 1)[2], col="black", lty="dotted")
  abline(v=which(df$epoch == 1)[3], col="black", lty="dotted")
  abline(v=which(df2$epoch == 1)[2], col="red", lty="dotted")
  abline(v=which(df2$epoch == 1)[3], col="red", lty="dotted")
  abline(v=which(df3$epoch == 1)[2], col="orange", lty="dotted")
  abline(v=which(df3$epoch == 1)[3], col="orange", lty="dotted") 
}

df = read.csv("output_stochastic_depth_resnet_new/long_baseline_more_augment_lr0.1_leto18.0.txt")
df2 = read.csv("output_stochastic_depth_resnet_new/long_depth_more_augment_lr0.1.0.txt")
df3 = read.csv("output_stochastic_depth_resnet_new/long_nonlinearity_more_augment_lr0.1.0.txt")

labels=c("baseline", "depth", "nonlinearity")
fills=c("black", "red", "orange")

plot(df$train_loss, type="l", ylab="train loss", xlab="epoch")
lines(df2$train_loss, col="red")
lines(df3$train_loss, col="orange")
plot_lr_lines()
legend("topright", legend=labels, fill=fills)

plot(df$avg_valid_loss, type="l", ylim=c(0,2), ylab="valid loss", xlab="epoch")
lines(df2$avg_valid_loss, col="red")
lines(df3$avg_valid_loss, col="orange")
plot_lr_lines()
legend("topright", legend=labels, fill=fills)

plot(df$valid_accuracy, type="l", ylim=c(0,1), ylab="valid accuracy", xlab="epoch")
lines(df2$valid_accuracy, col="red")
lines(df3$valid_accuracy, col="orange")
plot_lr_lines()
legend("topright", legend=labels, fill=fills)

dev.off()