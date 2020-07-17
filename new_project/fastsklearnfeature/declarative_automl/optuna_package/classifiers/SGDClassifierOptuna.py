from sklearn.linear_model import SGDClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class SGDClassifierOptuna(SGDClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SGDClassifier_')
        self.loss = trial.suggest_categorical(self.name + "loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"])
        self.penalty = trial.suggest_categorical(self.name + "penalty", ["l1", "l2", "elasticnet"])
        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-7, 1e-1)
        self.l1_ratio = trial.suggest_loguniform(self.name + "l1_ratio", 1e-9, 1)
        self.fit_intercept = True
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.epsilon = trial.suggest_loguniform(self.name + "epsilon", 1e-5, 1e-1)
        self.learning_rate = trial.suggest_categorical(self.name + "learning_rate", ["optimal", "invscaling", "constant"])
        self.eta0 = trial.suggest_loguniform(self.name + "eta0", 1e-7, 1e-1)
        self.power_t = trial.suggest_uniform(self.name + "power_t", 1e-5, 1)
        self.average = trial.suggest_categorical(self.name + "average", [False, True])

        #todo: add conditional parameters
