from sklearn.decomposition import TruncatedSVD

class TruncatedSVDOptuna(TruncatedSVD):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'TruncatedSVD_'

        self.target_dim = trial.suggest_int(self.name + "target_dim", 10, 256)

