from diffprivlib.models import LogisticRegression
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class PrivateLogisticRegressionOptuna(LogisticRegression):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('PrivateLogisticRegression_')

        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.C = trial.suggest_loguniform(self.name + "C", 0.03125, 32768)
        self.fit_intercept = True

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PrivateLogisticRegression_')
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node)
        space_gen.generate_number(self.name + "C", 1.0, depending_node=depending_node)
