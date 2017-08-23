binom.var = function(p, k=5) {
  return(k*p*(1-p))
}


p = seq(from=0,to=1,by=0.05)

plot(x=p, y=binom.var(p,k=10), type="l")
