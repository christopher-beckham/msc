get.fxs = function(fx, num.classes) {
  num.classes=4
  c = 1:num.classes
  cf = factorial(c)
  return(c*log(fx) - fx - log(cf))
}

softmax = function(fx, tau=1) {
  res = exp(fx / tau) / sum(exp(fx / tau))
  return(res)
}

steps = seq(from=0.1,to=5,by=0.25)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x),1.0)
  barplot(res,names.arg=1:4)
}

steps = seq(from=0.1,to=5,by=0.25)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x),0.3)
  barplot(res,names.arg=1:4)
}

steps = seq(from=0.1,to=5,by=0.25)
par(mfrow=c(4,5))
for(x in steps) {
  res = softmax(get.fxs(x),0.1)
  barplot(res,names.arg=1:4)
}
