fc = read.csv("combinedforecasts.csv")
combineddata = read.csv("combineddata.csv")
forecasts = fc[-1,-1] 
cvmse = t(fc[1,-1])

#Bates-Granger weights
bgw = cvmse^-1/sum(cvmse^-1)

#Bates-Granger combined forecast
bgcombo = as.matrix(forecasts) %*% as.vector(bgw)
bgcombo_mse = mean((bgcombo-combineddata[145:178,2])^2)

