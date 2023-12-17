library(xgboost)
library(caret)
library(randomForest)

combineddata = read.csv("combineddata.csv")
train <- combineddata[2:144,-1]
test <- combineddata[145:178,-1]

xgbtrain <- xgb.DMatrix(data = as.matrix(train)[,-1], label = train$sg20y)

par <- list(booster = "gbtree", objective = "reg:squarederror", eta = 0.1, gamma = 0,
  max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1)

#######################################
#Exploration of xgboost using 5-fold CV
#######################################

set.seed(5)
xgbcv <- xgb.cv(params = par, data = xgbtrain, nrounds = 200, nfold = 5, showsd = T, 
  stratified = T, print_every_n = 10, maximise = FALSE)

##################################
#Tuning hyperparameters of xgboost
##################################

xgbGrid = expand.grid(
  max_depth = 1:10,
  eta = c(0.1, 0.15, 0.2, 0.25, 0.3),
  subsample = 1:10*0.1,
  nrounds = 1:150,
  min_child_weight = 1, colsample_bytree = 1, gamma = 0
)

xgbcontrol = trainControl(method = "cv", number = 5, search = "grid")

set.seed(5)

model = train(y = train$sg20y, x = train[,2:59], method = "xgbTree", 
  objective = "reg:squarederror", 
  trControl = xgbcontrol, tuneGrid = xgbGrid)

xgbparameters = model$bestTune

xgb_cvmse = mean(model$resample$RMSE)^2
cat("CV MSE of gradient boost is", xgb_cvmse)

xgb_imp = varImp(model)
xgb_imp10 = xgb_imp
xgb_imp10$importance = data.frame(Overall = xgb_imp$importance[1:10,])
rownames(xgb_imp10$importance) = rownames(xgb_imp$importance)[1:10]
ggplot(xgb_imp10)

######################################
#Expanding window forecast for xgboost
######################################

forecastpar <- as.list(c(booster = "gbtree", objective = "reg:squarederror", xgbparameters))

xgb_oos = data.frame("pred" = 1:34*0)

for(i in 144:177){
  windowtrain = combineddata[2:i,-1]
  windowtest = combineddata[i+1,-1]
  xgbtrainwindow <- xgb.DMatrix(data = as.matrix(windowtrain[,-1]), label = windowtrain$sg20y)
  xgbtestwindow <- xgb.DMatrix(data = as.matrix(windowtest[,-1]), label = windowtest$sg20y)
  
  set.seed(5)
  xgboost = xgb.train(
    params = forecastpar,
    nrounds = forecastpar$nrounds,
    data = xgbtrainwindow,
  )
  
  xgb_oos$pred[i-143] = predict(xgboost, xgbtestwindow)
}

cat("Test MSE using eXtreme Gradient Boosting: ", RMSE(xgb_oos$pred, test$sg20y)^2)


########################################
#Tuning hyperparameters of random forest
########################################
set.seed(5)
rftune = data.frame(ntree = 5:20*0, mtry = 5:20*0, cvmse = 5:20*0)

for(x in 5:20){
  set.seed(5)
  rfmodel = train(y = train$sg20y, x = train[,2:59], method = "rf", ntree = x*100, trControl = xgbcontrol)
  
  rftune$ntree[x-4] = x*100
  rftune[x-4,] = c(x*100, rfmodel$bestTune$mtry, mean(rfmodel$resample$RMSE))
}

cat("Parameters chosen for Random Forest: ntree = ", rftune[which.min(rftune$cvmse),1], ", mtry = ", rftune[which.min(rftune$cvmse),2])


########################
#CV MSE of Random Forest
########################
set.seed(5)
model_rf = train(y = train$sg20y, x = train[,2:59], method = "rf", ntree = 900,
  trControl = xgbcontrol)

cat("CV MSE of random forest is ", mean(model_rf$resample$RMSE^2))


rf_imp = varImp(model_rf)
rf_imp10 = rf_imp
rf_imp10$importance = data.frame(Overall = rf_imp$importance[1:10,])
rownames(rf_imp10$importance) = rownames(rf_imp$importance)[1:10]
ggplot(rf_imp10)

############################################
#Expanding window forecast for random forest
############################################

rf_oos = data.frame("pred" = 1:34*0)

for(i in 144:177){
  windowtrain = combineddata[2:i,-1]
  windowtest = combineddata[i+1,-1]
  
  set.seed(5)
  rftrain = randomForest(x = windowtrain[,-1], y = windowtrain$sg20y, ntree = 900, mtry = 31)
  rf_oos$pred[i-143] = predict(rftrain, windowtest)
}

cat("Test MSE using Random Forest: ", RMSE(rf_oos$pred, test$sg20y)^2)
