from diffprivlib.models import GaussianNB
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class PrivateGaussianNBOptuna(GaussianNB):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('PrivateGaussianNB_')
        self.var_smoothing = trial.suggest_loguniform(self.name + "var_smoothing", 1e-11, 1e-7)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PrivateGaussianNB_')
        space_gen.generate_number(self.name + "var_smoothing", 1e-9, depending_node=depending_node)
