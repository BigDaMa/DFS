from sklearn.neighbors import KNeighborsClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class KNeighborsClassifierOptuna(KNeighborsClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('KNeighborsClassifier_')
        self.n_neighbors = trial.suggest_int(self.name + "n_neighbors", 1, 100, log=True)
        self.weights = trial.suggest_categorical(self.name + "weights", ["uniform", "distance"])
        self.p = trial.suggest_categorical(self.name + "p", [1, 2])
