### Boosted Sparse Nonlinear Metric Learning
### Auxillary function: Calculate the D array using local neighborhoods
### Author: Yuting Ma 
### Date: 04/27/2015


cal.D <- function(X, y, k, local=T, bag.fraction) {
  
  ### X = input variable matrix
  ### y = class label (-1 = neg, 1 = pos)
  ### k.pos, k.neg = number of local nearest neighbors used in pos and neg classes
  ### If use local information, set k = k.pos= k.neg small; 
  ### when using the entire sample, set k.pos= n.class.pos, k.neg=n.class.neg
  
  n <- nrow(X)
  p <- ncol(X)
  n.class <- c(sum(y==-1), sum(y==1))
  dist.X <- as.matrix(dist(X, diag=T, upper=T))
  S <- matrix(rep((0.5*y + 0.5), n),n,n,byrow=T) #pos.class=1, neg.class=0
  S.pos <- (1-S)*999 + S*dist.X  #neg.class=999, pos.class=original dist
  S.neg <- S*999 + (1-S)*dist.X #pos.class=999, neg.class=original dist
  diag(S.pos) <- diag(S.neg) <- rep(999,n) # set self-to-self dist to 999
  # Indices of nearest pos.class and nearest neg.class
  D <- array(dim=c(p,p,n))
  dimnames(D)[1] <- dimnames(D)[2] <- list(paste("X", 1:p, sep=""))
  if(local == T){
    pos.ind <- matrix(apply(S.pos, 1, function(x) order(x)[1:k]),k,n)
    neg.ind <- matrix(apply(S.neg, 1, function(x) order(x)[1:k]),k,n)
    for(i in 1:n){
      neg.sum <- rowSums(apply(X[neg.ind[,i],],1,function(x)(X[i,]-x) %*% t(X[i,]-x) ))
      pos.sum <- rowSums(apply(X[pos.ind[,i],],1,function(x)(X[i,]-x) %*% t(X[i,]-x) ))
      D[,,i] <- (1/k)*matrix(neg.sum - pos.sum, p,p)
    }
  } else{
    pos.ind <- neg.ind <- NULL
    for(i in 1:n){
      neg.sum <- rowSums(apply(X[y==-1,],1,function(x)(X[i,]-x) %*% t(X[i,]-x) ))
      pos.sum <- rowSums(apply(X[y==1,],1,function(x)(X[i,]-x) %*% t(X[i,]-x) ))
      D[,,i] <- matrix(neg.sum/n.class[1] - pos.sum/n.class[2], p,p)
    }
  }
  return(list(D=D, pos.ind=pos.ind, neg.ind=neg.ind))
}
