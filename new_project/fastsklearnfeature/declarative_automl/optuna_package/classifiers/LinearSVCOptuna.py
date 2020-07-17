from sklearn.svm import LinearSVC
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class LinearSVCOptuna(LinearSVC):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('LinearSVC_')
        self.penalty = trial.suggest_categorical(self.name + "penalty", ["l1", "l2"])
        self.loss = trial.suggest_categorical(self.name + "loss", ["hinge", "squared_hinge"])
        self.dual = False

        # This is set ad-hoc
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.C = trial.suggest_loguniform(self.name + "C", 0.03125, 32768)
        self.multi_class = "ovr"
        # These are set ad-hoc
        self.fit_intercept = True
        self.intercept_scaling = 1

        #todo: add conditional parameters
        if self.dual == False:
            if self.penalty == 'l2':
                self.loss = "squared_hinge"

        if self.penalty == 'l1':
            self.loss = "squared_hinge"
