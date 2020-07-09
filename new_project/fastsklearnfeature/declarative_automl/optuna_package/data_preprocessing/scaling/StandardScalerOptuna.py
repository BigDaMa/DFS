from sklearn.preprocessing import StandardScaler

class StandardScalerOptuna(StandardScaler):
    def init_hyperparameters(self, trial, X, y):
        pass
