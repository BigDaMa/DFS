from sklearn.linear_model import PassiveAggressiveClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class PassiveAggressiveOptuna(PassiveAggressiveClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('PassiveAggressive_')
        self.C = trial.suggest_loguniform(self.name + "C", 1e-5, 10)
        self.fit_intercept = True
        self.loss = trial.suggest_categorical(self.name + "loss", ["hinge", "squared_hinge"])
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.average = trial.suggest_categorical(self.name + "average", [False, True])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PassiveAggressive_')

        space_gen.generate_number(self.name + "C", 1.0, depending_node=depending_node)
        space_gen.generate_cat(self.name + "loss", ["hinge", "squared_hinge"], "hinge", depending_node=depending_node)
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node)
        space_gen.generate_cat(self.name + "average", [False, True], False, depending_node=depending_node)

