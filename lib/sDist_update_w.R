### Boosted Sparse Nonlinear Metric Learning
### Auxillary function: Update weight applied on each rank-one matrix by using bi-section searching
### Author: Yuting Ma 
### Date: 08/27/2014

update.w <- function(g, r, y, w.l=0, w.u=10, epsilon=1e-04) {
  
  ### wl, wu = prespecified lower/upper bound of weight w
  
  while(w.u - w.l > epsilon){
    w <- 0.5 * (w.l + w.u)
    if (sum(r*g* exp(-w*g*y)) > 0){
      w.l <- w
    } else {
      w.u <- w
    }
  }
  return(w)
}
