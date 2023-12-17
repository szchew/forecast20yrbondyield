library(nnet)
library(caret)
combineddata <- read.csv("combineddata.csv", stringsAsFactors = TRUE)

#Removing NA, min-max standardizing data
data = combineddata[2:178,-1]
train = data[1:143,]
test = data[144:177,]

maxs = apply(train, 2, max)
mins = apply(train, 2, min)

trains = as.data.frame(scale(train, center = mins, scale = maxs - mins))
tests = as.data.frame(scale(test, center = mins, scale = maxs - mins))


### Gauge relationship of decay rate with test mse (Fixed Window)

dec=c(0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10)
nd=length(dec) #no of lambdas, will determine no of iterations in a loop

nnmse=NULL #blank for collecting MSE for different lambdas

set.seed(5)
for(i in 1:nd) {
  nn=nnet(sg20y~., data=trains, 
          size=10,  maxit=1000, 
          decay=dec[i], linout = TRUE, 
          trace=FALSE)
  
  # Predict the test values and un-normalize 
  ## the predictions (don't forget!)
  yhat=predict(nn, tests)*(maxs[1]-mins[1])+mins[1]
  
  #Compute the test set MSE
  nnmse[i]=summary(lm((yhat-test$sg20y)^2~1))$coef[1]
}

result = data.frame(Lambda = dec, MSE = nnmse)
cat("Best performing decay rate: ", result$Lambda[which.min(result$MSE)])

##### Tuning hyperparameters with caret
nnetgrid = expand.grid(size = 1:15, decay = c(0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10))

set.seed(5)
nnetcv <- train(sg20y~., data = trains,
  method = "nnet", trControl = trainControl(method = "cv", number = 5),
  linout = TRUE, maxit = 1000, trace = FALSE, tuneGrid = nnetgrid
  )

cat("CV MSE of hypertuned ANN: ", mean((nnetcv$resample$RMSE*(maxs[1]-mins[1]))^2))

##### Expanding window forecast for hypertuned ANN

df <- data
df_predict <- data.frame(Index=numeric(0),sg20y_true=numeric(0),sg20y_predict=numeric(0),
                         residual = numeric(0))
for (i in 143:176) {
  train.ew <- df[1:i,] # Creeping training size
  
  maxew = apply(train.ew, 2, max) 
  minew = apply(train.ew, 2, min)
  train.ews = as.data.frame(scale(train.ew, center = minew, scale = maxew - minew))
  
  set.seed(5)
  nn=nnet(sg20y~., data=train.ews, 
          size=8,  maxit=1000, 
          decay=0.1, linout = TRUE, 
          trace=FALSE)

  #1-step ahead forecast
  test.ew <- df[i+1,]
  test.ews = as.data.frame(scale(test.ew, center = minew, scale = maxew - minew))
  forecast <- predict(nn, test.ews)*(maxew[1]-minew[1])+minew[1]
  
  true_sg20y <- test.ew$sg20y # Single test data
  
  df_predict[nrow(df_predict) + 1,] <- c(i+1,true_sg20y, forecast, forecast - true_sg20y)
}

cat("1-step-ahead MSE of ANN: ", mean(df_predict$residual^2))

