library(leaps)
library(caret)

rm(list=ls()) #use this to clear the variables

# Import SG20y interest rate data
df <- read.csv("combineddata.csv", header=TRUE, stringsAsFactors=FALSE)
head(df)

#Remove rows that contain at least one missing value
df = na.omit(df)

#Set the training set size:
trsize=143

#Generate trsize numbers from 1 to trsize in running order
trindices = 1:trsize

#get the training sample (getting only rows whose indices are in trindices)
train = df[trindices,]

#get the testing sample (remove rows whose indices are in trindices)
test = df[-trindices,]

fml = formula(sg20y~.-Month)

test_mat = model.matrix(fml, data = test)

######################
## FORWARD STEPWISE ##
######################
maxregno_bfs = 58
bfs = regsubsets(fml, data = train, nvmax = maxregno_bfs, method = "forward")
bfs_sum = summary(bfs)

#Finding BIC and AIC for "variance estimate from the model"
#Information criteria 
kbic_for=which.min(bfs_sum$bic) #BIC choice
kaic_for=which.min(bfs_sum$cp)  #AIC choice (AIC proportional to Cp)

#Finding BIC and AIC for "iterative variance" when when P=61 is relative large w.r.t N = 143
i = 1 #i denotes number of iteration
varYbic_for = var(train$sg20y) #sample variance of Y
varYaic_for = var(train$sg20y) #sample variance of Y
kbic_temp_for = 0; kaic_temp_for = 0
repeat{
  if (i > 1){ ## For iteration after the first, we need to update the error variance
    varYbic_for = bfs_sum$rss[kbic_iter_for]/(trsize-kbic_iter_for-1) 
    varYaic_for = bfs_sum$rss[kaic_iter_for]/(trsize-kaic_iter_for-1)
    ### NOTE: (trsize-kaic_iter_for-1) is the degree of freedom 
    #### (accounted for how many variables are involved)
  }
  
  bic_iter_for = bfs_sum$rss/trsize + 
    log(trsize)*varYbic_for*((1:maxregno_bfs)/trsize)
  aic_iter_for = bfs_sum$rss/trsize + 
    2*varYaic_for*((1:maxregno_bfs)/trsize) 
  #Select best models
  kbic_iter_for = which.min(bic_iter_for)
  kaic_iter_for = which.min(aic_iter_for)
  if (kbic_temp_for == kbic_iter_for & kaic_temp_for == kaic_iter_for){
    break #when convergence occurs, stop the loop
  }
  #store this iteration s' choice of k to compare with the new ones later.
  kbic_temp_for = kbic_iter_for
  kaic_temp_for = kaic_iter_for
  i = i +1
}

cat('Forward stepwise selection','\n', 
  'BIC best k: ',kbic_for ,'\n',
  'AIC best k: ',kaic_for ,'\n',
  'BIC best k - variance from iterative variance: ',kbic_iter_for,'\n',
  'AIC best k - variance from iterative variance: ',kaic_iter_for,'\n' )

#BIC CV MSE using caret
chosen_coefs_exp = coef(bfs, id = kbic_for)
bic_vars_exp = names(chosen_coefs_exp)[2:length(chosen_coefs_exp)]

set.seed(5)
bic_for_cv <- train(
  as.formula(paste("sg20y ~", paste(bic_vars_exp, collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)
cat("CV MSE of BIC model = ", mean(bic_for_cv$resample$RMSE^2))

#BIC Expanding Window (Forward)
forbpred = (143:176*0)

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(bic_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_vars_exp = names(lm_chosen_coefs_exp)
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  forbpred[i-142] = predicted_exp
}
bic_mse_for_exp_3 = mean(unlist(mse_store_exp))

cat("Out-of-sample MSE of FSS model chosen by BIC: ", bic_mse_for_exp_3)

#AIC CV MSE using caret
chosen_coefs_exp = coef(bfs, id = kaic_for)
aic_vars_exp = names(chosen_coefs_exp)[2:length(chosen_coefs_exp)]

set.seed(5)
aic_for_cv <- train(
  as.formula(paste("sg20y ~", paste(aic_vars_exp, collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)
cat("CV MSE of AIC model = ", mean(aic_for_cv$resample$RMSE^2))


#AIC Expanding Window (Forward)
forapred = 143:176*0

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(aic_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_vars_exp = names(lm_chosen_coefs_exp)
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  forapred[i-142] = predicted_exp
}
aic_mse_for_exp_14 = mean(unlist(mse_store_exp))

cat("OOS MSE of FSS model chosen by AIC: ", aic_mse_for_exp_14)

#AIC (iterative variance) CV MSE
chosen_coefs_exp = coef(bfs, id = kaic_iter_for)
aic_vars_exp = names(chosen_coefs_exp)[2:length(chosen_coefs_exp)]

set.seed(5)
aic_for_cv <- train(
  as.formula(paste("sg20y ~", paste(aic_vars_exp, collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

cat("CV MSE of AIC (iterative variance) model = ", mean(aic_for_cv$resample$RMSE^2))

#AIC (iterative variance) Expanding Window (Forward)
foraivpred = 143:176*0

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(aic_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_vars_exp = names(lm_chosen_coefs_exp)
  lm_chosen_coefs_exp[is.na(lm_chosen_coefs_exp)] <- 0 # account for NA values in coeff
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  foraivpred[i-142] = predicted_exp
}

aic_mse_back_exp_13 = mean(unlist(mse_store_exp))

cat("OOS MSE for BSS chosen by AIC (iterative variance): ", aic_mse_back_exp_13)

## 5 - FOLD CV
Kfold=5 #number of folds
set.seed(5)

one_fifth = rep(1:5, trsize%/%5+1)
ran_ind = sample(1:trsize, trsize, replace = FALSE) 
group_cv = one_fifth[ran_ind]

#get training matrix
trainmat = model.matrix(fml, data=train)
li_for = list()
li_for_exp = list()

mse_mat_for_cv = matrix(0,Kfold,50) #matrix to MSE results

for(fold in 1:Kfold) {
  fold_test = group_cv == fold #TRUE for those belong to test fold (only 1 fold used as test)
  folds_train = group_cv != fold #TRUE for those belong to train folds
  bss_train_fold_for = regsubsets(fml, data = train[folds_train,], nvmax = maxregno_bfs, method = "forward")

  for(k in 1:50) { #note: k is the number of variables used in the model
    bss_coef_fold_for = coef(bss_train_fold_for, id = k)
    var_sel_fold_for = trainmat[fold_test,names(bss_coef_fold_for)]  #only those variables chosen by bic
    fold_pred_for = var_sel_fold_for%*%bss_coef_fold_for #Xb' to get yhat
    mse_mat_for_cv[fold, k] = mean((train$sg20y[fold_test]-fold_pred_for)^2)
  }
  
}

bestK_cv = which.min(colMeans(mse_mat_for_cv)) #figure out best K (number of regressors)
cat("Best K chosen by CV MSE: ", bestK_cv)

chosen_coefs_cv_for = coef(bfs,id = bestK_cv)
print.table(chosen_coefs_cv_for)

# CV FSS expanding window
cv_vars_exp = names(chosen_coefs_cv_for)[-1]
forcvpred = 143:176*0

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(cv_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_chosen_coefs_exp[is.na(lm_chosen_coefs_exp)] <- 0 # account for NA values in coeff
  lm_vars_exp = names(lm_chosen_coefs_exp)
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  forcvpred[i-142] = predicted_exp
}

cat("OOS MSE for FSS chosen by 5-fold CV: ", mean(unlist(mse_store_exp)))

#CV MSE for CV-chosen model using caret
set.seed(5)
cv_for_cv <- train(
  as.formula(paste("sg20y ~", paste(names(chosen_coefs_cv_for)[-1], collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)
cat("CV MSE of CV-chosen model = ", mean(cv_for_cv$resample$RMSE^2))


#######################
## BACKWARD STEPWISE ##
#######################
maxregno_bbs = 58
bbs = regsubsets(fml, data = train, nvmax = maxregno_bbs, method = "backward", really.big = TRUE)
bbs_sum = summary(bbs)

#Finding BIC and AIC for "variance estimate from the model"
#Information criteria 
kbic_back=which.min(bbs_sum$bic) #BIC choice
kaic_back=which.min(bbs_sum$cp)  #AIC choice (AIC proportional to Cp)

#Finding BIC and AIC for "iterative variance" when when P=61 is relative large w.r.t N = 143
i = 1 #i denotes number of iteration
varYbic_back = var(train$sg20y) #sample variance of Y
varYaic_back = var(train$sg20y) #sample variance of Y
kbic_temp_back = 0; kaic_temp_back = 0
repeat{
  if (i > 1){ ## For iteration after the first, we need to update the error variance
    varYbic_back = bbs_sum$rss[kbic_iter_back]/(trsize-kbic_iter_back-1) 
    varYaic_back = bbs_sum$rss[kaic_iter_back]/(trsize-kaic_iter_back-1)
    ### NOTE: (trsize-kaic_iter_for-1) is the degree of freedom 
    #### (accounted for how many variables are involved)
  }
  
  bic_iter_back = bbs_sum$rss/trsize + 
    log(trsize)*varYbic_back*((1:maxregno_bbs)/trsize)
  aic_iter_back = bbs_sum$rss/trsize + 
    2*varYaic_back*((1:maxregno_bbs)/trsize) 
  #Select best models
  kbic_iter_back = which.min(bic_iter_back)
  kaic_iter_back = which.min(aic_iter_back)
  if (kbic_temp_back == kbic_iter_back & kaic_temp_back == kaic_iter_back){
    break #when convergence occurs, stop the loop
  }
  #store this iteration s' choice of k to compare with the new ones later.
  kbic_temp_back = kbic_iter_back
  kaic_temp_back = kaic_iter_back
  i = i +1
}

cat('Backward stepwise selection','\n', 
  'BIC best k: ',kbic_back ,'\n', 
  'AIC best k: ',kaic_back ,'\n',
  'BIC best k - variance from iterative variance: ',kbic_iter_back,'\n',
  'AIC best k - variance from iterative variance: ',kaic_iter_back,'\n')

#BIC CV MSE
chosen_coefs_exp = coef(bbs, id = kbic_back)
bic_vars_exp = names(chosen_coefs_exp)[2:length(chosen_coefs_exp)]

set.seed(5)
bic_back_cv <- train(
  as.formula(paste("sg20y ~", paste(bic_vars_exp, collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

cat("CV MSE of BIC model = ", mean(bic_back_cv$resample$RMSE^2))

#BIC Expanding Window (Backward)
backbpred = 143:176*0

chosen_coefs_exp = coef(bbs, id = kbic_back)
bic_vars_exp = names(chosen_coefs_exp)[2:length(chosen_coefs_exp)]

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(bic_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_vars_exp = names(lm_chosen_coefs_exp)
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  backbpred[i-142] = predicted_exp
}
bic_mse_back_exp = mean(unlist(mse_store_exp))

cat("OOS MSE for BSS chosen by BIC: ", bic_mse_back_exp)

#AIC CV MSE using caret
chosen_coefs_exp = coef(bbs, id = kaic_back)
aic_vars_exp = names(chosen_coefs_exp)[2:length(chosen_coefs_exp)]

set.seed(5)
aic_back_cv <- train(
  as.formula(paste("sg20y ~", paste(aic_vars_exp, collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

cat("CV MSE of AIC model = ", mean(aic_back_cv$resample$RMSE^2))

#AIC Expanding Window (Backward)
backaicpred = 143:176*0

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(aic_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_vars_exp = names(lm_chosen_coefs_exp)
  lm_chosen_coefs_exp[is.na(lm_chosen_coefs_exp)] <- 0 # account for NA values in coeff
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  backaicpred[i-142] = predicted_exp
}
aic_mse_back_exp_12 = mean(unlist(mse_store_exp))

cat("OOS MSE for BSS chosen by AIC: ", aic_mse_back_exp_12)

#AIC (iterative variance) CV MSE
chosen_coefs_exp = coef(bbs, id = kaic_iter_back)
aic_vars_exp = names(chosen_coefs_exp)[2:length(chosen_coefs_exp)]

set.seed(5)
aic_back_cv <- train(
  as.formula(paste("sg20y ~", paste(aic_vars_exp, collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

cat("CV MSE of AIC (iterative variance) model = ", mean(aic_back_cv$resample$RMSE^2))

#AIC (iterative variance) Expanding Window (Backward)
backaicivpred = 143:176*0

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(aic_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_vars_exp = names(lm_chosen_coefs_exp)
  lm_chosen_coefs_exp[is.na(lm_chosen_coefs_exp)] <- 0 # account for NA values in coeff
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  backaicivpred[i-142] = predicted_exp
}
aic_mse_back_exp_13 = mean(unlist(mse_store_exp))

cat("OOS MSE for BSS chosen by AIC (iterative variance): ", aic_mse_back_exp_13)

## 5 - FOLD CV
Kfold=5 #number of folds
set.seed(5)

one_fifth = rep(1:5, trsize%/%5+1)
ran_ind = sample(1:trsize, trsize, replace = FALSE) 
group_cv = one_fifth[ran_ind]

#get training matrix
trainmat = model.matrix(fml, data=train)

mse_mat_back_cv = matrix(0,Kfold,50) #matrix to MSE results
li_back = list()
li_back_exp = list()

for(fold in 1:Kfold) {
  fold_test = group_cv == fold #TRUE for those belong to test fold (only 1 fold used as test)
  folds_train = group_cv != fold #TRUE for those belong to train folds
  bss_train_fold_back = regsubsets(fml, data = train[folds_train,], nvmax = maxregno_bbs, method = "backward")
  
  for(k in 1:50) { #note: k is the number of variables used in the model
    bss_coef_fold_back = coef(bss_train_fold_back, id = k)
    var_sel_fold_back = trainmat[fold_test,names(bss_coef_fold_back)]  #only those variables chosen by bic
    fold_pred_back = var_sel_fold_back%*%bss_coef_fold_back #Xb' to get yhat
    mse_mat_back_cv[fold, k] = mean((train$sg20y[fold_test]-fold_pred_back)^2)
    }
}

bestK_cv_back = which.min(colMeans(mse_mat_back_cv)) #figure out best K (number of regressors)

cat("Best K chosen by CV MSE: ", bestK_cv_back)

chosen_coefs_cv_back = coef(bbs,id=bestK_cv_back)
print.table(chosen_coefs_cv_back)

#CV Expanding Window Forecast (Backward)
cv_vars_exp = names(chosen_coefs_cv_back)[2:length(chosen_coefs_cv_back)]
backcvpred = 143:176*0

mse_store_exp = list()
for(i in 143:176){
  lm_lin_exp = lm(paste("sg20y ~", paste(cv_vars_exp, collapse="+")), data = df[1:i,])
  lm_chosen_coefs_exp = coef(lm_lin_exp)
  lm_vars_exp = names(lm_chosen_coefs_exp)
  lm_chosen_coefs_exp[is.na(lm_chosen_coefs_exp)] <- 0 # account for NA values in coeff
  predicted_exp = test_mat[i-142,lm_vars_exp]%*%lm_chosen_coefs_exp
  truevals_exp = test$sg20y[i-142]
  mse_store_exp = append(mse_store_exp,(truevals_exp-predicted_exp)^2)
  backcvpred[i-142] = predicted_exp
  }

cat("OOS MSE for BSS chosen by 5-fold CV: ", mean(unlist(mse_store_exp)))

#CV MSE for CV-chosen model using caret
set.seed(5)
cv_back_cv <- train(
  as.formula(paste("sg20y ~", paste(names(chosen_coefs_cv_back)[-1], collapse="+"))), data = train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)
cat("CV MSE of CV-chosen model = ", mean(cv_back_cv$resample$RMSE^2))
