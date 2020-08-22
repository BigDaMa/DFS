from sklearn.neighbors import KNeighborsClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class KNeighborsClassifierOptuna(KNeighborsClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('KNeighborsClassifier_')
        self.n_neighbors = trial.suggest_int(self.name + "n_neighbors", 1, min([100, X.shape[0]]), log=True)
        self.weights = trial.suggest_categorical(self.name + "weights", ["uniform", "distance"])
        self.p = trial.suggest_categorical(self.name + "p", [1, 2])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('KNeighborsClassifier_')
        space_gen.generate_number(self.name + "n_neighbors", 1, depending_node=depending_node)
        space_gen.generate_cat(self.name + "weights", ["uniform", "distance"], "uniform", depending_node=depending_node)
        space_gen.generate_cat(self.name + "p", [1, 2], 2, depending_node=depending_node)
