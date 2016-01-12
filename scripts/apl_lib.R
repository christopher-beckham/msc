apl = function(as, bs) {
  fn = function(x) {
    res = max(0, x)
    for(s in 1:length(as)) {
      res = res + ( as[s] * max(0, -x + bs[s]) )
    }
    return(res)
  }
  return(fn)
}