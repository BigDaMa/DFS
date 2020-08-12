from sklearn.svm import SVC
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class SVCOptuna(SVC):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SVC_')
        self.C = trial.suggest_loguniform(self.name + "C", 0.03125, 32768)
        self.kernel = trial.suggest_categorical(self.name + "kernel", ["rbf", "poly", "sigmoid"])
        if self.kernel == "poly":
            self.degree = trial.suggest_int(self.name + "degree", 2, 5, log=False)

        self.gamma = trial.suggest_loguniform(self.name + "gamma", 3.0517578125e-05, 8)

        if self.kernel == "poly" or self.kernel == "sigmoid":
            self.coef0 = trial.suggest_uniform(self.name + "coef0", -1, 1)
        self.shrinking = trial.suggest_categorical(self.name + "shrinking", [True, False])
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.max_iter = -1

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SVC_')

        space_gen.generate_number(self.name + "C", 1.0, depending_node=depending_node)
        category_kernel = space_gen.generate_cat(self.name + "kernel", ["rbf", "poly", "sigmoid"], "rbf", depending_node=depending_node)

        space_gen.generate_number(self.name + "degree", 3, depending_node=category_kernel[1])
        space_gen.generate_number(self.name + "gamma", 0.1, depending_node=depending_node)
        space_gen.generate_number(self.name + "coef0", 0, depending_node=depending_node) #fix if we change to graph
        space_gen.generate_cat(self.name + "shrinking", [True, False], True, depending_node=depending_node)
        space_gen.generate_number(self.name + "tol", 1e-3, depending_node=depending_node)