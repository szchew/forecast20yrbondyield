library(caret)
library(ggplot2)
library(tseries)
library(reshape2)
combineddata = read.csv("combineddata.csv")

####################
#Exploration of Data
####################
pacf(na.omit(combineddata$sg20y), lag.max = 5, main = "SG 20Y yields")
adf.test(na.omit(combineddata$sg20y), k = 1)

#ADF test on first-differenced
adf.test(na.omit(combineddata$sg20y - combineddata$sg20yl1), k = 1)

cormat = cor(na.omit(combineddata[c(2,41:60)]))
melted_cormat <- melt(cormat)

ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
    midpoint = 0, limit = c(-1,1), space = "Lab") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
    size = 10, hjust = 1), axis.title.x = element_blank(), axis.title.y = element_blank())+
  coord_fixed() + 
  ggtitle("Yield and Macro Factors")

train <- combineddata[2:144,-1]
test <- combineddata[145:178,-1]

############
#AR(1) Model
############

ar(train$sg20y, aic = TRUE, order.max = 5, method = "ols")
ar1_oos = data.frame("pred" = 1:34*0)

#Rolling window forecast
for(i in 144:177){
  ar1 = ar(combineddata$sg20y[1:i], aic = FALSE, order.max = 1, method = "yw")
  ar1_oos$pred[i-143] = predict(ar1)$pred[1]
}

cat("AR(1) test MSE: ", RMSE(ar1_oos$pred, test$sg20y)^2)

##################
#Random walk model
##################
rw_oos = test$sg20yl1

cat("RW test MSE: ", RMSE(rw_oos, test$sg20y)^2)


