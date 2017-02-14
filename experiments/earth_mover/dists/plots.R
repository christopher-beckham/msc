dfx = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists//dr_xent_l2-1e-4_sgd_pre_split_hdf5.modelv1.44.bak2.csv",header=F)
dfe = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_exp_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak4.csv",header=F)
dfemd2 = read.csv("~/Desktop/lisa_tmp4_4/msc/experiments/earth_mover/dists/dr_emd2_l2-1e-4_sgd_pre_split_hdf5.modelv1.23.bak2.csv",header=F)


barplot(height=as.numeric(df[1,1:5]))

# we want to find dists where the correct class was predicted
# correctly, AND the correct class is not zero

is.correct = function(vector, correct_class) {
  return( (which.max(vector)-1) == correct_class )
}

plot.stuff = function(df) {
  par(mfrow=c(3,3))
  nonzero = which(df[,6] != 0 )
  nc=0
  limit=9
  for(i in 1:length(nonzero)) {
    # if we didn't get this example correctly predicted via argmax, skip it
    #if( is.correct(df[nonzero[i],1:5], df[nonzero[i],6]) == FALSE ) {
    #  next
    #}
    print(paste("processing #", nonzero[i]))
    barplot(height=as.numeric(df[nonzero[i],1:5]),names.arg=0:4,main=df[nonzero[i],6])
    nc = nc + 1
    if( nc == limit) {
      break
    }
  }  
}

pdf("dfx.pdf",height=5,width=5)
plot.stuff(dfx)
dev.off()

pdf("dfe.pdf",height=5,width=5)
plot.stuff(dfe)
dev.off()

# ----------

g.like = function(arr, which_idx) {
  b1 = all( sort(arr[j:length(arr)],decreasing=T) == arr[j:length(arr)] )
  b2 = all( arr[1:j] == sort(arr[1:j]) )
  return(all(c(b1,b2)))
}

get.g.like = function(df, num_classes=5) {
  qs = c()
  for( q in 1:nrow(df)) {
    # if g-like
    if( g.like( as.numeric(df[q,1:num_classes]), df[q,num_classes+1]+1) ) {
      # if it is also correct
      #if( is.correct( df[q,1:num_classes], df[q,num_classes+1] ) ) {
      qs = c(qs, q)
      #}
    }
  }
  return(qs)
}


