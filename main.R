### Boosted Sparse Nonlinear Metric Learning
### Example Main.R: an example of implementing sDist algorithm on Madelon data with cross-validation
### Author: Yuting Ma 
### Date: 04/27/2015

setwd("~") # set working directory to the main folder of sDist
output_path <- "./output/"

load("./data/madelon_dat.RData")
source("./lib/sDist_algorithm.R")
X <- madelon_dat$X
y <- madelon_dat$y
data_name <- "madelon"

capOrder_values <- c(1, 2)
bag_values <- c(0.1, 0.3, 0.5, 0.7)
step_values <- c(10, 50, 999)
rho_values <- c(0.05, 0.1, 0.2)

capOrder_values <- 1
bag_values <- c(0.1, 0.3, 0.5, 0.7)
step_values <- 10
rho_values <- 0.1

tune_values <- expand.grid(cap.order=capOrder_values, bag.fraction=bag_values, update.step=step_values, 
                           rho = rho_values)

# choose the combination of tuning parameters
comp_id <- 1
tune_val <- tune_values[comp_id,]

M <- 499
lambda <- c(0, 0.1, 1, 10)
local <- T
verbose <- T
shrinkage <- 0.01
k <- 3
mtry <- NULL

cap.order <- tune_val$cap.order
bag.fraction <- tune_val$bag.fraction
update.step <- tune_val$update.step
rho <- tune_val$rho

args <- list(rho=rho, shrinkage=shrinkage, cap.order=cap.order,
             bag.fraction=bag.fraction, update.step=update.step, mtry=mtry)

cat(paste(names(args), args, collapse="_", sep="_"), "\n")

fold <- 5  
n_sample <- nrow(X)
num.cls0.fold <- rep(260, 5)  # number of class-0 point in each fold
num.cls1.fold <- rep(260, 5) # number of class-1 point in each fold
assign_cls1 <- rep(1:fold, times=num.cls1.fold)
assign_cls0 <- rep(1:fold, times=num.cls0.fold)

CV_index <- rep(NA, n_sample)
CV_index[y == 1] <- sample(assign_cls1, length(assign_cls1))
CV_index[y == -1] <- sample(assign_cls0, length(assign_cls0))
fit_cv <- rep(NA, n_sample)
for(c in 1:fold){
  test_id <- which(CV_index == c)
  X_train <- X[-test_id,]
  y_train <- y[-test_id]
  X_test <- X[test_id,]
  y_test <- y[test_id]
  model <- sDist(X_train, y_train, X_test, y_test, M=M, lambda=lambda, local=T, verbose=T, rho=rho, shrinkage=shrinkage, 
                   cap.order=cap.order, bag.fraction=bag.fraction, update.step=update.step, mtry=mtry, k=k, save.list=F)
  fit_cv[test_id] <- model$pred
}

cat("CV error =", round(mean(fit_cv != y), digits=4), "\n")
