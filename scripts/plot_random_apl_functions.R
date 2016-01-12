source("apl_lib.R")

args = commandArgs(trailingOnly = TRUE)
S = 1
if( length(args) != 0 ) {
  S = as.numeric(args[1])
}

xs = seq(from=-5,to=5,by=0.01)
X11()
par(mfrow=c(2,2))
for(i in 1:4) {
  plot( x=xs, y=sapply(xs, apl( rnorm(S,0,1), rnorm(S,0,1)) ), type="l" )
}

locator()