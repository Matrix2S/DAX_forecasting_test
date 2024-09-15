# DAX_forecasting_test
In this project I tried recreating a time-series forecasting model introduced by C. N. Babu and B. E. Reddy in 2015. (Babu, C. N., &amp; Reddy, B. E. (2015). Performance comparison of four new ARIMA-ANN prediction models on internet traffic data. Journal of telecommunications and information technology, (1), 67-75.) The idea is to first split the data in two time-series by preprocessing it with an Moving Average Filter. Then run an ARIMA model on one part and a neural network model on the other. The final forecast is made by adding both forecasts together. 

I analysed their model and methods during my bachelor's thesis and saw it as an opportunity to test my python coding skills for Data Science. 
Their model is not to complex but also gives me enough room to work with different python data analysis(science) libraries.

The task is to build a ARIMA-NeuralNetwork Hybrid model to predict DAX-30 closing values, for which I used daily DAX-30 data from 2000 - 2023. For simplyfication I used an AR(p) model for the ARIMA(p,d,q) model and a for the nueral network I used an Multilayer Perceptron with **one** hiddenlayers of size 40 to reduce computation.

GDAXI.csv - Daily DAX-30 data from 2000-2023 with closing/ opening values, daily volume, etc.
MA_Filter.py - Pre-processing step to the input data
ARIMA_forecast.py - Get the ARIMA(p,d,q) forecast for the input data (in this case its a AR(p) model)
MLP_forecast.py - Get the neural network forecast for the input data (in this case its a multi layer perceptron)
Model_building.py - Recreate the model proposed in (C. N. Babu & B. E. Reddy, 2015) using the above tools

Libraries used: numpy, pandas, sklearn, statsmodels, matplotlib, math
