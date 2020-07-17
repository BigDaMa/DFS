from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class FastICAOptuna(FastICA):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('FastICA_')

        self.n_components = trial.suggest_int(self.name + "n_components", 10, 2000, log=False)
        self.algorithm = trial.suggest_categorical(self.name + 'algorithm', ['parallel', 'deflation'])
        self.whiten = trial.suggest_categorical(self.name + 'whiten', [False, True])
        self.fun = trial.suggest_categorical(self.name + 'fun', ['logcosh', 'exp', 'cube'])

