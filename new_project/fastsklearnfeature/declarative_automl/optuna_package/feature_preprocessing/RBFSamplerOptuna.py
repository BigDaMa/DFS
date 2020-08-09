from sklearn.kernel_approximation import RBFSampler
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class RBFSamplerOptuna(RBFSampler):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('RBFSampler_')

        self.gamma = trial.suggest_loguniform(self.name + "gamma", 3.0517578125e-05, 8)
        self.n_components = trial.suggest_int(self.name + "n_components", 50, 10000, log=True)

    def generate_hyperparameters(self, space_gen):
        self.name = id_name('RBFSampler_')

        space_gen.generate_number(self.name + "gamma", 1.0)
        space_gen.generate_number(self.name + "n_components", 100)

