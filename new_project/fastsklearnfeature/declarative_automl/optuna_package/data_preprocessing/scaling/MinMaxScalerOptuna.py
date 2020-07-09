from sklearn.preprocessing import MinMaxScaler

class MinMaxScalerOptuna(MinMaxScaler):
    def init_hyperparameters(self, trial, X, y):
        pass
