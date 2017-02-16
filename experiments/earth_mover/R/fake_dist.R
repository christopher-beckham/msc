labels = c("baby","kid","schooler","teen","adult","senior","elder")

cols = rep("#1f77b4", length(labels))
cols[5] = "#ff7f03"

dist1 = c(5,6,14,10,20,3,1)
dist1 = dist1 / sum(dist1)

# bad, because schooler > teen
barplot(dist1,names.arg=labels,las=2,col=cols,ylab="p(y|x)",main="distribution A")

dist2 = c(14,6,5,10,20,3,1)
dist2 = dist2 / sum(dist2)

# swap baby with schooler, now it's even worse
barplot(dist2,names.arg=labels,las=2,col=cols,ylab="p(y|x)",main="distribution B")

dist3 = c(5,6,8,10,20,3,1)
dist3 = dist3 / sum(dist3)

# this is the best because it's g-like
barplot(dist3,names.arg=labels,las=2,col=cols, main="distribution C")

pdf("dist_abc.pdf", width=5, height=2)
par(mfrow=c(1,3))
barplot(dist1,names.arg=labels,las=2,col=cols,ylab="p(y|x)",main="distribution A")
barplot(dist2,names.arg=labels,las=2,col=cols,ylab="p(y|x)",main="distribution B")
barplot(dist3,names.arg=labels,las=2,col=cols, main="distribution C")
dev.off()




