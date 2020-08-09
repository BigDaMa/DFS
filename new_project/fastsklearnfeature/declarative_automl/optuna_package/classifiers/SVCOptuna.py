from sklearn.svm import LinearSVC
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class SVCOptuna(LinearSVC):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SVC_')
        self.C = trial.suggest_loguniform(self.name + "C", 0.03125, 32768)
        self.kernel = trial.suggest_categorical(self.name + "kernel", ["rbf", "poly", "sigmoid"])
        self.degree = trial.suggest_int(self.name + "degree", 2, 5, log=False)
        self.gamma = trial.suggest_loguniform(self.name + "gamma", 3.0517578125e-05, 8)
        self.coef0 = trial.suggest_uniform(self.name + "coef0", -1, 1)
        self.shrinking = trial.suggest_categorical(self.name + "shrinking", [True, False])
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.max_iter = -1

        #todo: add conditional parameters

    def generate_hyperparameters(self, space_gen):
        self.name = id_name('SVC_')

        space_gen.generate_number(self.name + "C", 1.0)
        space_gen.generate_cat(self.name + "kernel", ["rbf", "poly", "sigmoid"], "rbf")
        space_gen.generate_number(self.name + "degree", 3)
        space_gen.generate_number(self.name + "gamma", 0.1)
        space_gen.generate_number(self.name + "coef0", 0)
        space_gen.generate_cat(self.name + "shrinking", [True, False], True)
        space_gen.generate_number(self.name + "tol", 1e-3)