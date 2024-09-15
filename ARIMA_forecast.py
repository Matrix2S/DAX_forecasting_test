import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class AR_forecast:
    def __init__(self,data, order, split):
        self.d     = data.dropna()
        self.d.index = range(0,len(self.d))
        self.order = order
        self.split = split
        
        self.forecast = []
        self.s_index = int(len(self.d)*self.split)+1
        self.d_train = self.d[:self.s_index]
        
    def ar_forecast(self):
        AR_1 = ARIMA(self.d_train, order=(self.order, 0, 0)).fit()
        
        self.forecast.append(float(AR_1.forecast())) # t=1 forecast
        for i in range(1,len(self.d[self.s_index:])):
            AR_1 = AR_1.extend(self.d[self.s_index+i-1:self.s_index+i])
            self.forecast.append(float(AR_1.forecast()))
            
    def get_forecast(self):
        return np.array(self.forecast)




