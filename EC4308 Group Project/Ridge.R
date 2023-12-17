library(glmnet)
library(caret)

## Importing and cleaning data
combineddata <- read_csv("combineddata.csv")
df = na.omit(combineddata)

X = model.matrix(sg20y~., data = df[,-1]) # TO REGRESS AGAINST EVERYTHING
X = X[, colnames(X)!= "(Intercept)"] # SINCE RIDGE RUNs ON DEMEANED DATA, DOESNT NEED INTERCEPT
Y = df$sg20y

## Train test split
trsize = 143 #training sample size
train = 1:trsize  #training set is not randomly chosen!!
test = -train

## Fit ridge on training set
set.seed(5)
ridge.mod <- glmnet(X[train,], Y[train], alpha = 0, thresh = 1e-12)
plot(ridge.mod,xlab="L2 norm")

sumsq_coef = c()
lambdas = c()
for (i in 1:ncol(coef(ridge.mod))){
  lambdas = c(lambdas, ridge.mod$lambda[i]) #get lambda value
  coefs = coef(ridge.mod)[-1,i] #get coefficients, excluding intercept
  sumsq_coef = c(sumsq_coef, sqrt(sum(coefs^2)))
}

plot(x = lambdas, y = sumsq_coef, xlim = c(0, 10000), 
#here we stop at 10000 but there are more values for lambda
  xlab = "values of lambdas",
  ylab = "sum of square roots of the coefficients")

##### CV using default grid
set.seed(5)
cv5fold = cv.glmnet(X[train,], Y[train], alpha = 0, nfolds = 5)
plot(cv5fold$lambda,cv5fold$cvm, #  manually pull out lambdas and MSE computed in this CV algorithm
  main="5-fold CV default settings", xlab="Lambda", ylab="CV MSE")

##### CV using user grid, since default grid chose the smallest lambda
#notice that for values higher than the value 0.05003107, the CV MSE seems to rise

ugrid = seq(0, 1, length.out = 10000) #grid that zooms in on promising intervals

set.seed(5)
cv5fold_ridge_ug = cv.glmnet(X[train,], Y[train], alpha = 0, lambda = ugrid, nfolds=5)
plot(cv5fold_ridge_ug)

lmin = cv5fold_ridge_ug$lambda.min
l1se = cv5fold_ridge_ug$lambda.1se

##### Expanding window forecast
df_predict <- data.frame(Index=numeric(0),sg20y_true=numeric(0),predict_min=numeric(0),
                         predict_1se = numeric(0), residmin = numeric(0), resid1se = numeric(0))

for (i in 143:176) {
  train.ew <- df[1:i,-1] # Creeping training size
  train.x.ew <-  model.matrix(sg20y ~ ., data = train.ew)[,-1] 
  train.y.ew <- train.ew$sg20y
  test.ew <- df[i+1,-1] 
  test.x.ew <- as.matrix(test.ew[,-1])
  true_sg20y <- test.ew$sg20y # Single test data
  
  set.seed(5)
  cv5fold.ew <- cv.glmnet(train.x.ew, train.y.ew, alpha=0, lambda = ugrid, nfolds = 5, type.measure = "mse")
  forecastmin <- predict(cv5fold.ew, newx=test.x.ew, s="lambda.min", type="response")
  forecast1se <- predict(cv5fold.ew, newx=test.x.ew, s="lambda.1se", type="response")
  
  df_predict[nrow(df_predict) + 1,] <- c(i+1,true_sg20y, forecastmin, forecast1se, 
    forecastmin - true_sg20y, forecast1se - true_sg20y)
} 

cat("Expanding window test MSE (lambda,min):", round(mean(df_predict$residmin^2),4),"\n",
  "Expanding window test MSE (lambda.1se):", round(mean(df_predict$resid1se^2),4))


##### CV MSE of models using caret
ridgegrid = expand.grid("lambda" = lmin, "alpha" = 0)
set.seed(5)
lambda_min_cv <- train(
  x = X[train,], y = Y[train], 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 5), tuneGrid = ridgegrid
)

ridgegrid1se = expand.grid("lambda" = l1se, "alpha" = 0)
set.seed(5)
lambda_1se_cv <- train(
  x = X[train,], y = Y[train], 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 5), tuneGrid = ridgegrid1se
)

cat("CV MSE of RIDGE (lambda.min) = ", mean(lambda_min_cv$resample$RMSE^2), "\n")
cat("CV MSE of RIDGE (lambda.se) = ", mean(lambda_1se_cv$resample$RMSE^2))