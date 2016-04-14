png("plot.png")
# bottom left top right
par(mar=c(5,4,1,2))
df1 = read.csv("output/vgg_a.txt")
plot(df1$train_loss, type="l", col="blue", xlab="# epochs", ylab="avg loss")
lines(df1$valid_loss, type="l", col="red" )
legend("topright", legend=c("train loss", "valid loss"), fill=c("blue", "red"))
dev.off()