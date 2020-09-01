from sklearn.decomposition import PCA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class PCAOptuna(PCA):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('PCA_')

        self.n_components = trial.suggest_uniform(self.name + "n_components", 0.5, 0.9999)
        self.whiten = trial.suggest_categorical(self.name + "whiten", [False, True])

        self.sparse = False

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PCA_')

        space_gen.generate_number(self.name + "n_components", 0.9999, depending_node=depending_node)
        space_gen.generate_cat(self.name + "whiten", [False, True], False, depending_node=depending_node)
