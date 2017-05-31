preset.cols = c("#1f77b4ff","#ff7f0eff","#2ca02cff","#d62728ff","#9467bdff","#8c564bff","e377c2ff")
preset.cols.alpha = c("#1f77b444","#ff7f0e44","#2ca02c44","#d6272844","#9467bd44","#8c564b44","e377c244")


get.loess = function(vc, span=0.08) {
  xx = 1:length(vc)
  return(predict(loess(vc ~ xx,span=0.08)))
}

do.plot = function(dfs, exp.names, exp.cols, exp.ltys, which.col, xlab, ylab, xlim, ylim) {
  for( i in 1:length(dfs)) {
    df = dfs[[i]]
    if(i==1) {
      plot(get.loess(df[which.col][,1]), type="l", lwd=1.5, col=exp.cols[i], lty=exp.ltys[i], xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab)
    } else {
      lines(get.loess(df[which.col][,1]), type="l", lwd=1.5, col=exp.cols[i], lty=exp.ltys[i], xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab)
    }
  }
  legend("bottomright", legend=exp.names, lwd=1.5, col=exp.cols, lty=exp.ltys, bty="n", cex=0.8)
}