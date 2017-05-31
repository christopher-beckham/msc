prep.margins = function() {
  par(mar=c(2,1,2.5,1)+0.1) 
}

pdf("binom_p_k4_tau1.pdf")
ps = seq(from=0,to=1,by=0.051)
num.classes=4
prep.margins()
par(mfrow=c(4,5))
for(p in ps) {
  res = dbinom( x=0:(num.classes-1), size=num.classes-1, prob=p)
  print(sum(res))
  barplot(res,names.arg=0:(num.classes-1),main=paste("p =",p))
}
dev.off()

pdf("binom_p_k8_tau1.pdf")
ps = seq(from=0,to=1,by=0.051)
num.classes=8
prep.margins()
par(mfrow=c(4,5))
for(p in ps) {
  res = dbinom( x=0:(num.classes-1), size=num.classes-1, prob=p)
  print(sum(res))
  barplot(res,names.arg=0:(num.classes-1),main=paste("p =",p))
}
dev.off()