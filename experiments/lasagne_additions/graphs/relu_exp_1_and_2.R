pdf(file="relu_exp_1_and_2.pdf")

par(mfrow=c(3,1))

df = read.csv("../output_stochastic_depth_resnet_new/long_baseline_more_augment_lr0.1_leto18.0.txt")
df_2 = read.csv(
  "/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/lasagne_additions/output_stochastic_depth_resnet_new/long_baseline_more_augment_lr0.1_leto04.1.fixed.txt"
)
min_rows = min( nrow(df), nrow(df_2) )
df = df[1:min_rows,]
df_2 = df_2[1:min_rows,]

df2 = read.csv("../output_stochastic_depth_resnet_new/long_depth_more_augment_lr0.1.0.txt")
df2_2 = read.csv(
  "/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/lasagne_additions/output_stochastic_depth_resnet_new/long_depth_more_augment_lr0.1.1.fixed.txt"
)
min_rows = min( nrow(df2), nrow(df2_2) )
df2 = df2[1:min_rows,]
df2_2 = df2_2[1:min_rows,]

df3 = read.csv("../output_stochastic_depth_resnet_new/long_nonlinearity_more_augment_lr0.1.0.txt")
df3_2 = read.csv(
  "/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/lasagne_additions/output_stochastic_depth_resnet_new/long_nonlinearity_more_augment_lr0.1.1.fixed.txt"
)
min_rows = min( nrow(df3), nrow(df3_2) )
df3 = df3[1:min_rows,]
df3_2 = df3_2[1:min_rows,]

df4 = read.csv(
"/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/lasagne_additions/output_stochastic_depth_resnet_new/long_baseline_more_augment_rrelu_lr0.1_leto18.0.txt"
)
df4_2 = read.csv(
  "/Users/cjb60/Desktop/lisa_tmp4_4/msc/experiments/lasagne_additions/output_stochastic_depth_resnet_new/long_baseline_more_augment_rrelu_lr0.1_leto18.1.txt"
)
min_rows = min( nrow(df4), nrow(df4_2) )
df4 = df4[1:min_rows,]
df4_2 = df4_2[1:min_rows,]

labels=c("baseline", "depth", "nonlinearity", "rrelu")
fills=c("black", "red", "orange", "purple")

# --------- TRAINING LOSS ---------

plot(df$train_loss, type="l", ylab="train loss", xlab="epoch")
lines(df_2$train_loss)
abline(v=171, col="black", lty="dotted")

lines(df2$train_loss, col="red")
lines(df2_2$train_loss, col="red")
abline(v=188, col="red", lty="dotted")

lines(df3$train_loss, col="orange")
lines(df3_2$train_loss, col="orange")
abline(v=168, col="orange", lty="dotted")

lines(df4$train_loss, col="purple")
lines(df4_2$train_loss, col="purple")
abline(v=224, col="purple", lty="dotted")

legend("topright", legend=labels, fill=fills)

# ----------------------------------

# --------- VALIDATION LOSS ---------

plot(df$avg_valid_loss, type="l", ylim=c(0.35,2), ylab="valid loss", xlab="epoch")
lines(df_2$avg_valid_loss)
abline(v=171, col="black", lty="dotted")

lines(df2$avg_valid_loss, col="red")
lines(df2_2$avg_valid_loss, col="red")
abline(v=188, col="red", lty="dotted")

lines(df3$avg_valid_loss, col="orange")
lines(df3_2$avg_valid_loss, col="orange")
abline(v=168, col="orange", lty="dotted")

lines(df4$avg_valid_loss, col="purple")
lines(df4_2$avg_valid_loss, col="purple")
abline(v=224, col="purple", lty="dotted")

legend("topright", legend=labels, fill=fills)

# ----------------------------------

# --------- VALIDATION ACCURACY ---------

plot(df$valid_accuracy, type="l", ylim=c(0.7,0.95), ylab="valid accuracy", xlab="epoch")
lines(df_2$valid_accuracy)
abline(v=171, col="black", lty="dotted")

lines(df2$valid_accuracy, col="red")
lines(df2_2$valid_accuracy, col="red")
abline(v=188, col="red", lty="dotted")

lines(df3$valid_accuracy, col="orange")
lines(df3_2$valid_accuracy, col="orange")
abline(v=168, col="orange", lty="dotted")

lines(df4$valid_accuracy, col="purple")
lines(df4_2$valid_accuracy, col="purple")
abline(v=224, col="purple", lty="dotted")

legend("topright", legend=labels, fill=fills)

# ----------------------------------

print(max(df$valid_accuracy))
print(max(df_2$valid_accuracy))
print("--")
print(max(df2$valid_accuracy))
print(max(df2_2$valid_accuracy))
print("--")
print(max(df3$valid_accuracy))
print(max(df3_2$valid_accuracy))
print("--")
print(max(df4$valid_accuracy))
print(max(df4_2$valid_accuracy))

dev.off()