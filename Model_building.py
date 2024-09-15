
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor
from MA_filter import Ma_Filter
from ARIMA_forecast import AR_forecast
from MLP_forecast import MLP_forecast


d = pd.read_csv("C:\\Users\\Salman\\Desktop\\PythonDataAnalysis\\DAX Forecast\\DAX Forecast\\GDAXI.csv")
d = d["Close"]

Filter = Ma_Filter(d)
Filter.ma_filter()
Rest = Filter.get_rest()
MA = Filter.get_filter()

MLP = MLP_forecast(MA, 40, 0.8)
MLP.mlp_forecast(size=(10, 2))

AR  = AR_forecast(Rest, 40, 0.8)
AR.ar_forecast()

pred = MLP.get_forecast() + AR.get_forecast()
pred = pd.Series(pred)
d = Filter.d
pred.index = d[len(d)-len(pred):].index

math.sqrt(((d[len(d)-len(pred):] - pred)**2).mean()) # ERROR of 181.45 Points on average

# Plotting True Values withs predictions
fig, axes = plt.subplots(figsize = (12,6))
axes.plot(pred.iloc[800:850], color = "red", linewidth = 1, label = "Prediction")
axes.plot(d[len(d)-len(pred):].iloc[800:850], color = "blue", linewidth = 1, label = "True")
axes.legend()

# Plotting MLP predictions to trend data
MLP_pred = pd.Series(MLP.get_forecast())
MLP_pred.index = MA[len(MA)-len(MLP_pred):].index
fig, axes = plt.subplots(figsize = (12,6))
axes.plot(MLP_pred, color = "red", linewidth = 1, label = "Prediction")
axes.plot(MA[len(MA)-len(MLP_pred):], color = "blue", linewidth = 1, label = "True")
axes.legend()

# Plotting AR predictions to high frequency movement
AR_pred = pd.Series(AR.get_forecast())
AR_pred.index = Rest[len(Rest)-len(AR_pred):].index
fig, axes = plt.subplots(figsize = (12,6))
axes.plot(AR_pred.loc[5150:5250], color = "red", linewidth = 1, label = "Prediction")
axes.plot(Rest[len(Rest)-len(AR_pred):].loc[5150:5250], color = "blue", linewidth = 1, label = "True")
axes.legend()                               # AR model mostly predicts tomorrow as today !!!!
