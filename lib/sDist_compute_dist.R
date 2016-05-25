### Boosted Sparse Nonlinear Metric Learning
### Auxillary function: Compute nearest neighbor orders based on distance with metric W
### Author: Yuting Ma 
### Date: 04/14/2015

compute_dist <- function(X, y, W){
  n <- nrow(X)
  n_pos <- sum(y == 1)
  n_neg <- sum(y == -1)
  L <- chol(W)
  dist.X <- as.matrix(dist(X%*%t(L), diag=T, upper=T))
  S <- matrix(rep((0.5*y + 0.5), n),n,n,byrow=T) #pos.class=1, neg.class=0
  S.pos <- (1-S)*99999 + S*dist.X  #neg.class=999, pos.class=original dist
  S.neg <- S*99999 + (1-S)*dist.X #pos.class=999, neg.class=original dist
  diag(S.pos) <- diag(S.neg) <- rep(99999,n) # set self-to-self dist to 999
  pos_order <- t(matrix(apply(S.pos, 1, function(x) order(x)[1:(n_pos-1)]),n_pos-1,n))   # each row indicates the index of positive nearest neighbors of X[i,] (in order)
  neg_order <- t(matrix(apply(S.neg, 1, function(x) order(x)[1:(n_neg-1)]),n_neg-1,n))
  return(list(pos_order=pos_order, neg_order=neg_order))
}