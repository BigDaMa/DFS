from sklearn.decomposition import PCA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class KernelPCAOptuna(PCA):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('PCA_')

        self.keep_variance = trial.suggest_uniform(self.name + "keep_variance", 0.5, 0.9999)
        self.whiten = trial.suggest_categorical(self.name + "whiten", [False, True])

    def generate_hyperparameters(self, space_gen):
        self.name = id_name('PCA_')

        space_gen.generate_number(self.name + "keep_variance", 0.9999)
        space_gen.generate_cat(self.name + "whiten", [False, True], False)
