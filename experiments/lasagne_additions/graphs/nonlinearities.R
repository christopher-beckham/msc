relu_exp = function(x, p) {
  return( p*relu(x) + (1-p)*x )
}

relu = function(x) {
  return(max(0, x))
}

leaky_relu = function(x, a) {
  if(x < 0) {
    return(a*x)
  } else {
    return(x)
  }
}

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

sigmoid = function(x) {
  return(1 / (1 + exp(-x)))
}

sigmoid_exp = function(x, p) {
  return( p*sigmoid(x) + (1-p)*x )
}

tanh_exp = function(x) {
  return(p*tanh(x) + (1-p)*x)
}

# plot elu
xs = seq(-10, 10, 0.1)
probs = c(0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
rainbows = rainbow(length(probs)+1)
plot(x=xs, y=sapply(xs,elu_exp,p=1), type="l", ylim=c(-10,10), col=rainbows[1], ylab="g(x)", xlab="x")
for(i in 1:length(probs)) {
  lines(x=xs, y=sapply(xs,elu_exp,p=probs[i]), type="l", col=rainbows[i+1])
}
legend("bottomright", legend=c(1.0, probs), fill=rainbows)


# plot relu expectation
xs = seq(-5, 5, 0.1)
plot(x=xs, y=sapply(xs,relu_exp,p=1), type="l", ylim=c(-5,5))
for(prob in c(0.9, 0.7, 0.5, 0.3, 0.1)) {
  lines(x=xs, y=sapply(xs,relu_exp,p=prob), type="l")
}

# plot tanh expectation
xs = seq(-5, 5, 0.1)
plot(x=xs, y=sapply(xs,tanh_exp,p=1), type="l", ylim=c(-5,5))
for(prob in c(0.9, 0.7, 0.5, 0.3, 0.1)) {
  lines(x=xs, y=sapply(xs,tanh_exp,p=prob), type="l")
}





plot(x=xs, y=sapply(xs,relu_exp,p=0.9), type="l", ylim=c(-5,5))
lines(x=xs, y=sapply(xs,leaky_relu,a=0.1), type="l", col="red", ylim=c(-5,5))
