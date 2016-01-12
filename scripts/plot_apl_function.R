source("apl_lib.R")

as = c(-0.3)
bs = c(0.2)
args = commandArgs(trailingOnly = TRUE)
if( length(args) != 0 ) {
  as = as.numeric(strsplit(args[1], ",")[[1]])
  bs = as.numeric(strsplit(args[2], ",")[[1]])
}

xs = seq(from=-5,to=5,by=0.01)
X11()
plot( x=xs, y=sapply(xs, apl(as, bs)), type="l" )

locator()