from sklearn.neighbors import KNeighborsClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class KNeighborsClassifierOptuna(KNeighborsClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('KNeighborsClassifier_')
        self.n_neighbors = trial.suggest_int(self.name + "n_neighbors", 1, 100, log=True)
        self.weights = trial.suggest_categorical(self.name + "weights", ["uniform", "distance"])
        self.p = trial.suggest_categorical(self.name + "p", [1, 2])

    def generate_hyperparameters(self, space_gen):
        self.name = id_name('KNeighborsClassifier_')
        space_gen.generate_number(self.name + "n_neighbors", 1)
        space_gen.generate_cat(self.name + "weights", ["uniform", "distance"], "uniform")
        space_gen.generate_cat(self.name + "p", [1, 2], 2)
