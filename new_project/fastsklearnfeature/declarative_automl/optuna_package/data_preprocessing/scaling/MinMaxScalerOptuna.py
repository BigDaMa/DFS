from sklearn.preprocessing import MinMaxScaler

class MinMaxScalerOptuna(MinMaxScaler):
    def init_hyperparameters(self, trial, X, y):
        pass

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass