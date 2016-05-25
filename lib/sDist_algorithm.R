### Boosted Sparse Nonlinear Metric Learning
### Main sDist algorithm
### Author: Yuting Ma 
### Date: 04/27/2015

sDist <- function(X, y, X_test=NULL, y_test=NULL, M, lambda, local=T, verbose=T, rho=0.1, shrinkage=1, 
                    cap.order=3, bag.fraction=0.5, update.step=999, mtry=NULL, k=3, save.list = T, data_name=NULL){
  
  ### Inputs
  ### X = input variable matrix (n*p)
  ### y = class labels (-1,1)
  ### X_test = input variable matrix of the testing data
  ###        if X_test = NULL, then no prediction is made
  ### y_test = class labels of the test set
  ### M = maximum number of iterations
  ### lambda = regularizing parameter on the overall complexity measure. lambda can be a vector of values
  ### local (T/F) = whether use local neighborhood. Default = TRUE
  ### verbose (T/F) = whether trace the results from each boosting step. Default = TRUE
  ### rho (0, 1) or (integer) = if rho in (0, 1), it is the ratio between kappa and p_subset for learning sparse rank-one update.
  ###                           if rho > 1 and it is an integer, then kappa = rho = the degree of sparsity in rank-one update
  ### shrinkage = a shrinkage parameter applied to each rank-one updae. Default = 1 (no shrinkage). Also known as learning rate or step-size reduction.
  ### cap.order = the maximum order of interaction in the hierarchical expansion. Default = 3 (cubic interaction)
  ### bag.fraction = the proportion of subsample used in each boosting step. Default=0.5. If bag.fraction =1, no bagging is performed
  ### update.step (integer) = the number of steps in a run before sDist update local neighborhoods according to the current W. Default = 999 (never update)
  ### mtry = Number of variables randomly sampled as candidates at each boosting step for stochastic gradient descent. 
  ###        If mtry = NULL, use the default value sqrt(p). 
  ###        If mtry = NA, then no stochastic feature selection is performed. 
  ###        If mtry in (0,1), then use the input as a proportion of the total number of features.
  ### k = number of nearest neighbors used in the classifier. Default = 3
  ### save.list (T/F) = whether to save a list of training weights, objective values, train errors and etc.
  
  library(tensor)
  args <- list(rho=rho, shrinkage=shrinkage, cap.order=cap.order,
               bag.fraction=bag.fraction, update.step=update.step, mtry=mtry)  
  source("./lib/sDist_eigen_sparse_power.r")
  source("./lib/sDist_D_local.R")
  source("./lib/sDist_compute_dist.R")
  source("./lib/sDist_update_w.R")

  
  n <- nrow(X)
  p <- ncol(X)
  n.class <- c(sum(y==-1), sum(y==1))
  X <- as.matrix(X)
  dimnames(X)[2] <- list(paste("X", 1:p, sep=""))
  W_adj <- 1e-06   # for adjust small negative eigenvalues due to numerical computation
  
  used_time <- rep(NA, M)
  Z <- list()   # store vectors of rank-one updates
  weights <- rep(NA, M) #store weights of each update
  Cp <- rep(NA,M) #store overall complexity of updated metric
  W <- array(0, dim=c(p,p)) #Initialization of distance metric
  f_W <- rep(0, n)
  obj.val <- rep(NA, M) #store value of objective function with each metric
  train_err <- rep(NA, M)   #store training error of proposed classifier using current W
  S <- NULL    # initial selected set
  add <- NULL
  X.add <- X
  d <- p    #initial size of candidate set
  
  if(bag.fraction == 1){
    D_original_output <- cal.D(X, y, k)
    D_original <- D_original_output$D
    pos.ind <- D_original_output$pos.ind
    neg.ind <- D_original_output$neg.ind
    ind_subsample <- 1:n
    D <- D_original
  }  
  
  Dist_output <- compute_dist(X, y, diag(p))
  pos_order <- Dist_output$pos_order
  neg_order <- Dist_output$neg_order
  
  r <- y   #first residuals without random ordering 
  
  #Boosting iterations
  for(m in 1:M){  
    cat("m= ",m, "\n")
    ### update X and D
    ### 3 parts:
    ### - new variables add to X
    ### - update NN
    ### - new X_subsample
    start_time <- proc.time()
    
    ### if Nearest-Neighborhood update is needed
    if(m %% update.step == 0){
      W_current <- array(0, dim=c(d, d))
      for(j in 1:(m-1)){
        zz <- as.vector(Z[[j]])
        d.zz <- 1:length(zz)
        W_current[d.zz,d.zz] <- W_current[d.zz,d.zz] + weights[j]*(zz %*% t(zz))
      }
      W_current <- W_current + W_adj*diag(d)
      
      Dist_output <- compute_dist(X.add, y, W_current)
      pos_order <- Dist_output$pos_order
      neg_order <- Dist_output$neg_order
    }
    
    ### Expanding the input feature matrix
    if(length(add) != 0 & cap.order > 1){
      #if new variables are added
      # Update the entire X if new variables are added
      S.ind <- which(colnames(X.add) %in% S)
      add.ind <- which(colnames(X.add) %in% add)
      temp.add <- apply(expand.grid(add.ind,S.ind)[-2,],1, function(x) X.add[,x[1]]*X.add[,x[2]])
      X.add <- cbind(X.add, scale(temp.add))
      colnames(X.add)[-(1:d)] <- apply(expand.grid(add.ind,S.ind)[-2,],1, 
                                       function(x) paste(colnames(X.add)[x[1]], colnames(X.add)[x[2]],sep=""))
      
      # Eliminate terms of X with order greater than the cap.order
      X.names <- dimnames(X.add)[[2]]
      X.order <- unlist(lapply(X.names, function(x)length(unlist(strsplit(x, "X")))))-1
      if(any(X.order > cap.order)){
        X.add <- X.add[,-which(X.order > cap.order)]
      }
    }
    d.add <- ncol(X.add) - d
    
    ### number of feature selected randomly at each boosting step
    if(is.null(mtry)){   # use the default sqrt(p)
      p_subset <- round(sqrt(d))
    } else if (is.na(mtry)){   
      p_subset <- ncol(X.add)
    } else if (mtry < 1){   # proportion
      p_subset <- round(d*mtry)
    } else{
      p_subset <- mtry
    }
    #cat("Partial selection of features. Using ", p_subset, " variables in each boosting step. \n")
    
    ### for stochastic gradient descent
    if (p_subset == ncol(X.add)){   ### no feature subsampling
      var_subset <- 1:ncol(X.add)
    } else{
      var_subset <- sample(1:ncol(X.add), size=p_subset)  # construct feature subset on the newly formed feature candidate set
    }  
    
    ### Update D matrices
    if(bag.fraction == 1){   # if no bagging is needed, only append additional columns to D (no need to update the entire D)
      var_D_add <- which(var_subset > d)   # d = ncol(D_previous). Only add the selected ones that are not in previous candidnate set 
      if(cap.order == 1){
        D <- D_original[var_subset, var_subset, ]
      } else {
        if(length(add) != 0 && d.add > 0){
          D.add <- array(dim=c(ncol(X.add),ncol(X.add),n))
          dimnames(D.add)[[1]] <- dimnames(D.add)[[2]] <- as.vector(dimnames(X.add)[[2]])
          for(i in 1:n){
            neg.sum.add <- rowSums(apply(X.add[neg.ind[,i],],1,
                                         function(x) (X.add[i,-(1:d)] - x[-(1:d)]) %*% t(X.add[i,]-x)))
            pos.sum.add <- rowSums(apply(X.add[pos.ind[,i],],1,
                                         function(x) (X.add[i,-(1:d)] - x[-(1:d)]) %*% t(X.add[i,]-x)))
            v.add <- (1/k)*matrix(neg.sum.add - pos.sum.add, nrow=d.add, ncol=ncol(X.add))
            D.add[,,i] <- cbind(rbind(D_original[,,i],v.add[,1:d]),t(v.add))
          }
          D_original <- D.add
          D <- D_original[var_subset, var_subset,]
        }  
      } 
    } else {  # if use subsampling
      ### Subsample bag.fraction*100% of X, sampled without replacement.
      set.seed(m)    #set.seed for replication
      ind_subsample <- sample(1:n, size = round(bag.fraction*n), replace=F)
      #Compute D based on X_subsample
      pos.ind <- t(apply(pos_order, 1, function(x) x[x %in% ind_subsample][1:k])) # for all n samples
      neg.ind <- t(apply(neg_order, 1, function(x) x[x %in% ind_subsample][1:k]))   
      D <- array(dim=c(p_subset, p_subset,length(ind_subsample)))
      X_temp <- X.add[,var_subset]
      for(j in 1:length(ind_subsample)){
        i <- ind_subsample[j]
        neg.sum <- rowSums(apply(as.matrix(X_temp[na.omit(neg.ind[i,]),]),1,function(x)(X_temp[i,]-x) %*% t(X_temp[i,]-x)))
        pos.sum <- rowSums(apply(as.matrix(X_temp[na.omit(pos.ind[i,]),]),1,function(x)(X_temp[i,]-x) %*% t(X_temp[i,]-x)))
        D[,,j] <- matrix(neg.sum/sum(!is.na(neg.ind[i,])) - pos.sum/sum(!is.na(neg.ind[i,])), p_subset, p_subset)
      }
      dimnames(D)[[1]] <- dimnames(D)[[2]] <- as.vector(dimnames(X.add)[[2]][var_subset])
    }
    d <- ncol(X.add)
    
    ############################################################
    #Computation of sparse vector
    A <- tensor(D, r[ind_subsample], 3,1)
    if(rho < 1){
      kappa <- max(round(rho* p_subset), 1)
    } else{
      kappa <- rho
    }
    z <- eigen_sparse_power(A, kappa)$x    
    g <- apply(D, 3, function(x) t(z) %*% x %*% z)
    temp <- y[ind_subsample]*g
    w <- shrinkage * update.w(g, r[ind_subsample], y[ind_subsample])
  
    r[ind_subsample] <- r[ind_subsample] * exp(-w*y[ind_subsample]*g)
    # input nonzero weight to the original feature space (d-dimensional)
    z_temp <- rep(0, d)
    z_temp[var_subset] <- z
    z <- z_temp
    # Update weight matrix: expanding dimensions of W and add the learned rank-one matirx
    I <- cbind(diag(d-d.add), matrix(0,d-d.add,d.add))
    #W <-  t(I) %*% W %*% I + w*(z %*% t(z)) + W_adj*diag(d) # update W, in case more variables are included in the candidate set. This step is super costly when d is getting large.   
    
    #Expanding feature space when new variables are selected
    a <- dimnames(X.add)[[2]][which(z!=0)]
    if(is.null(S)){ add <- a} else {add <- a[which(!a %in% S)]}
    S <- unique(c(S,a)) #aggregate the selected set    
    
    #Recording
    Z <- c(Z, list(z))
    weights[m] <- w
    Cp[m] <- length(unique(unlist(sapply(Z, function(x) which(x != 0)))))
    f_W[ind_subsample] <- f_W[ind_subsample] + w*g
    predict <- ifelse(f_W>0,1,-1)
    obj.val[m] <- sum(exp(-y*f_W))
    train_err[m] <-  mean(predict != y) 
    
    # Tracing result 
    if (verbose){
      cat("   variable selected= ", a, "\n")
      cat("   weight=", w, "\n")
      cat("   obj.val= ",obj.val[m], "\n")     
      cat("   mean train error =", train_err[m], "\n")
    }
    
    end_time <- proc.time()
    used_time[m] <- (end_time - start_time)[3]
    if(save.list){
      save_list <- list(args=args, Z=Z, weights=weights, used_time=used_time, obj_values=obj.val,
                        train_err = train_err, names_X_add=colnames(X.add))
      save(save_list, file=paste(output_path, data_name,"_", paste(args, collapse="_") ,"_sDist_output.RData", sep=""))
    }
  }
  
  cat("kappa =", kappa, ", cap order= ", cap.order, ", bag.fraction =",bag.fraction, ", update.step= ", update.step, "\n")
  cat("mininum train error found at: m= ", which.min(train_err), " with err =", min(train_err), "\n")
  cat("mean used time =", mean(used_time), "\n")
  

  if(is.null(X_test)){
    return(save_list)
  }
  ####################################################################
  if(!is.null(X_test)){
    ### Make final predictions on the entire sample
    # Expand X_test to the same dimension as X_train
    if(d > p){
      interaction_term <- lapply(dimnames(X.add)[[2]], function(x) as.numeric(unlist(strsplit(x, "X"))[-1]))[(p+1):d]
      product <- function(x){Reduce("*", x)}   # to perform element-wise multiplication over a list of multiple vectors
      X_test.add <- cbind(X_test, do.call(cbind, lapply(interaction_term, function(i) product(lapply(i, function(j) X_test[,j])))))  
    } else{
      X_test.add <- X_test
    }
    dimnames(X_test.add)[[2]] <- dimnames(X.add)[[2]]
    
    
    err_est <- array(dim=c(4, length(lambda)))
    rownames(err_est) <- c("train_err_knn", "test_err_knn", "train_err_fW", "test_err_fW")
    colnames(err_est) <- paste("Lambda=", lambda, sep="")
    for (l in 1:length(lambda)){
      # Select the metric with the minimum overall loss
      final.ind <-lambda[l]
      final.d <- length(Z[[final.ind]])
      final.W <- matrix(0,final.d,final.d)
      for(m in 1:final.ind){
        zz <- as.vector(Z[[m]])
        d.zz <- 1:length(zz)
        final.W[d.zz,d.zz] <- final.W[d.zz,d.zz] + weights[m]*(zz %*% t(zz))
      }
      final.W <- final.W + W_adj*diag(final.d)
      colnames(final.W) <- rownames(final.W) <- colnames(X.add)[1:final.d]
      spar_degree <- Cp[final.ind]/d
      cat("Lambda =", lambda[l], "optimal stop=", final.ind,", spar_degree = ", spar_degree, "\n")
      
      X.final <- X.add[,1:final.d]
      X_test.final <- X_test.add[,1:final.d]
      dimnames(X_test.final)[[2]] <- dimnames(X.final)[[2]]
      select.final <- colnames(X.final)[which(rowSums(final.W) != 0)]
      # Leave-one-out knn prediction for training set
      L_final <- chol(final.W)
      dist_all <- as.matrix(dist(as.matrix(rbind(X.final,X_test.final))%*%t(L_final), diag=T, upper=T))
      diag(dist_all) <- 99999
      nn_ind <-apply(dist_all[,1:n], 1, function(x) order(x)[1:k])   # nearest neighbors in train data
      pred_knn <- apply(nn_ind, 2, function(i) as.numeric(names(which.max(table(y[i])))))
      # Prediction using f_W in sDist classifier
      pos_mean_dist <- apply(dist_all[, y==1], 1, function(x) mean((sort(x)[1:k])^2))
      neg_mean_dist <- apply(dist_all[, y==-1], 1, function(x) mean((sort(x)[1:k])^2))
      pred_fW <- sapply(neg_mean_dist - pos_mean_dist, function(x) ifelse(x>0, 1, -1))
      
      err_est[1,l] <- round(mean(pred_knn[1:n] != y), digits=3)
      err_est[2,l] <- round(mean(pred_knn[-(1:n)] != y_test), digits=3)
      err_est[3,l] <- round(mean(pred_fW[1:n] != y), digits=3)
      err_est[4,l] <- round(mean(pred_fW[-(1:n)] != y_test), digits=3)
    }
    return(list(pred = pred_fW[-(1:n)],  err_est=err_est, mean_time=mean(used_time)))
   }
}

