from sklearn.random_projection import GaussianRandomProjection
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class GaussianRandomProjectionOptuna(GaussianRandomProjection):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('GaussianRandomProjection_')

        self.eps = trial.suggest_loguniform(self.name + "eps", 1e-7, 1.0)


    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('GaussianRandomProjection_')

        space_gen.generate_number(self.name + "eps", 0.1, depending_node=depending_node)




