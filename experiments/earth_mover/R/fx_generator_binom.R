prep.margins = function() {
  par(mar=c(2,1,2.5,1)+0.1) 
}

pdf("binom_p_k4_tau1.pdf",height=4,width=6)
ps = seq(from=0,to=1,by=0.051)
num.classes=4
par(mfrow=c(4,5))
prep.margins()
for(p in ps) {
  res = dbinom( x=0:(num.classes-1), size=num.classes-1, prob=p)
  print(sum(res))
  barplot(res,names.arg=0:(num.classes-1),main=paste("p =",p))
}
dev.off()

pdf("binom_p_k8_tau1.pdf",height=4,width=6)
ps = seq(from=0,to=1,by=0.051)
num.classes=8
par(mfrow=c(4,5))
prep.margins()
for(p in ps) {
  res = dbinom( x=0:(num.classes-1), size=num.classes-1, prob=p)
  print(sum(res))
  barplot(res,names.arg=0:(num.classes-1),main=paste("p =",p))
}
dev.off()

