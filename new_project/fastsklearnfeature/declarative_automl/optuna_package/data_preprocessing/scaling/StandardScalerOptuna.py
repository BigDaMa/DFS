from sklearn.preprocessing import StandardScaler

class StandardScalerOptuna(StandardScaler):
    def init_hyperparameters(self, trial, X, y):
        pass

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass