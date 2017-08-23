get.fxs = function(fx, num.classes=4) {
  c = 1:num.classes
  cf = factorial(c)
  return(c*log(fx) - fx - log(cf))
}

softmax = function(fx, tau=1) {
  res = exp(fx / tau) / sum(exp(fx / tau))
  return(res)
}

prep.margins = function() {
  par(mar=c(2,1,2.5,1)+0.1) 
}

pdf("fx_k4_tau1-repeat.pdf",height=4,width=6)
steps = seq(from=0.1,to=5,by=0.25)
par(mfrow=c(4,5))
prep.margins()
for(x in steps) {
  res = softmax(get.fxs(x),1.0)
  barplot(res,names.arg=1:4,main=paste("f(x) =",x),cex.main=0.9)
}
dev.off()

pdf("fx_k4_tau0.3.pdf")
steps = seq(from=0.1,to=5,by=0.25)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x),0.3)
  barplot(res,names.arg=1:4,main=paste("f(x) =",x))
}
dev.off()

steps = seq(from=0.1,to=5,by=0.25)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x),0.1)
  barplot(res,names.arg=1:4)
}


# ------------------------------------

# 8 classes and tau=1.0
steps = seq(from=1,to=10,by=0.5)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x, num.classes=8),1.0)
  barplot(res,names.arg=1:8,main=x)
}

# 8 classes and tau=0.3
steps = seq(from=1,to=10,by=0.5)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x, num.classes=8),0.3)
  barplot(res,names.arg=1:8,main=x)
}

# -----------------------------------

# 20 classes and tau=1.0
steps = seq(from=1,to=20,by=1)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x, num.classes=20),0.5)
  barplot(res,names.arg=1:20,main=x)
}


