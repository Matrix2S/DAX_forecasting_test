import numpy as np
from sklearn.neural_network import MLPRegressor

class MLP_forecast:
    def __init__(self,data, order, split):
        self.d     = data.dropna()
        self.d.index = range(0,len(self.d))
        self.order = order
        
        self.s_index = int(len(self.d)*split)+1
        self.d_train = self.d[:self.s_index]
        
        self.train_y = np.array(self.d_train[self.order:self.s_index])
        self.train_x = [list(self.d_train[self.order-1:self.s_index-1])]
        for i in np.arange(1,self.order):
            self.train_x.append(list(self.d_train[self.order-1-i:self.s_index-1-i]))
        self.train_x = np.array(self.train_x).transpose()
        
        self.d_test  = [list(self.d[self.s_index-1:-1])]
        for i in np.arange(1,self.order):
            self.d_test.append(list(self.d[self.s_index-1-i:-1-i]))
        self.d_test = np.array(self.d_test).transpose()
        
    def mlp_forecast(self, size):
        self.MLP_1 = MLPRegressor(hidden_layer_sizes=size, max_iter=1000).fit(self.train_x, self.train_y)
            
    def get_forecast(self):
        MLP_1_pred = np.array(self.MLP_1.predict(self.d_test))
        return MLP_1_pred





