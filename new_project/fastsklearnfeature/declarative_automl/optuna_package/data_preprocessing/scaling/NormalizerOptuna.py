from sklearn.preprocessing import Normalizer

class NormalizerOptuna(Normalizer):
    def init_hyperparameters(self, trial, X, y):
        pass
