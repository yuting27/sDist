### Boosted Sparse Nonlinear Metric Learning
### Auxilliary function: Truncated power method for computing the sparse eigenvector asso.with largest eigenvalue
### Author: Yuting Ma 
### Date: 04/27/2015

eigen_sparse_power <- function(A, kappa, maxIter=50, optTol=1e-4){
  
  # Truncated power method for computing the sparse eigenvector asso.with largest eigenvalue
  # this method is used in sDist for computing the sparse \xi with given degree of sparsity k
  # Objective function: max x^T A x 
  #   				 s.t  ||x|| = 1, ||x||_0 = kappa
  # Input: 
  #	- A is p * p symmetric, not necessarily PSD, p is large and A is sparse
  #	- kappa = target level of sparsity of x
  #	- maxIter = max number of iterations (default = 50)
  #	- optTol = optimality tolerance (default = 1e-6)
  # Output: a
  #	- x = the sparse eigenvector assoc. with the largest eigenvalue with degree of sparsity kappa
  #	- f = objective function value at x = x^T A x
  
  library(Matrix)
  p = nrow(A)
  #Initialization
  x_0 = rep(0, p)
  x_0 <- truncate_operator(diag(A), kappa)
  x = as(x_0, "sparseMatrix")
  
  # Power step
  s = A %*% x
  g = 2*s
  f = as.numeric(t(x) %*% s)
  x_t = truncate_operator(g, kappa)
  f_old = f
  
  i <- 1
  
  #Main algorithmic loop
  while(i <= maxIter){
    s_t = A %*% x_t 
    f_t = as.numeric(t(x_t) %*% s_t)
    f = f_t
    
    # To ensure PSD: objective to be non-decreasing
    lambda = 1e-4
    j = 0
    while(f < f_old - 1e-10 && j <= maxIter){
      g_t = g + 2*lambda*x
      x_t = truncate_operator(g_t, kappa)
      s_t = A %*% x_t
      f_t = as.numeric(t(x_t) %*% s_t)
      f = f_t
      lambda = lambda * 10
      j <- j+1
    }
    
    if(abs(f - f_old) < optTol){
      break
    }
    x = x_t
    g = 2*s_t
    x_t = truncate_operator(g, kappa)
    f_old = f
    i = i + 1
  }
  return(list(f=f, x=as(x,'matrix')))	
}

### perform the truncation that keeps only the largest k absolute values in a vector
truncate_operator <- function(v, kappa){
  idx <- order(abs(v), decreasing=T)[1:kappa]
  u = rep(0, length(v))
  u[idx] = v[idx]
  u = u/sqrt(sum(u^2))	# normalization
  u = as(u, "sparseMatrix")
  return(u)
}


