elu = function(x) {
  if(x > 0) {
    return(x)
  } else {
    return (exp(x)-1)
  }
}

elu_exp = function(x, p) {
  return( p*elu(x) + (1-p)*x )
}

relu_exp = function(x, p) {
  return( p*relu(x) + (1-p)*x )
}

relu = function(x) {
  return(max(0, x))
}

pdf("elus_and_leaky_relus.pdf", height=4)

par(mfrow=c(1,2))

# plot relu
xs = seq(-10, 10, 0.1)
probs = c(0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
rainbows = rainbow(length(probs)+1)
plot(x=xs, y=sapply(xs,relu_exp,p=1), type="l", ylim=c(-10,10), col=rainbows[1], xlab="x", ylab="p*relu(x) + (1-p)x")
for(i in 1:length(probs)) {
  lines(x=xs, y=sapply(xs,relu_exp,p=probs[i]), type="l", col=rainbows[i+1])
}
legend("bottomright", legend=c(1.0, probs), fill=rainbows)

# plot elu
xs = seq(-10, 10, 0.1)
probs = c(0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
rainbows = rainbow(length(probs)+1)
plot(x=xs, y=sapply(xs,elu_exp,p=1), type="l", ylim=c(-10,10), col=rainbows[1], xlab="x", ylab="p*elu(x) + (1-p)x")
for(i in 1:length(probs)) {
  lines(x=xs, y=sapply(xs,elu_exp,p=probs[i]), type="l", col=rainbows[i+1])
}
legend("bottomright", legend=c(1.0, probs), fill=rainbows)

dev.off()