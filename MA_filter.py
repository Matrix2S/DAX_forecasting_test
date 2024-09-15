import pandas as pd

class Ma_Filter:
    def __init__(self, data):
        self.ma = 0
        self.l  = 0
        self.w  = 2
        self.MA = pd.Series([])
        self.L  = pd.Series([])
        self.d  = data.dropna()
        self.d.index = range(0,len(self.d))
    
    def ma_filter(self):
        while not 2.9 < self.ma < 3.1 and not 2.9 < self.l < 3.1:
            self.MA = self.d.rolling(window = self.w).mean()
            self.L = self.d - self.MA
            
            self.ma = (((self.MA - self.MA.mean())/(self.MA.std()))**4).mean()
            self.l  = (((self.L - self.L.mean())/(self.L.std()))**4).mean()
            
            self.w += 1
    
    def get_filter(self):
        return self.MA
    
    def get_rest(self):
        return self.L
    
    def get_filterlength(self):
        return self.w
    
    def get_kurt(self):
        return [self.ma, self.l]

