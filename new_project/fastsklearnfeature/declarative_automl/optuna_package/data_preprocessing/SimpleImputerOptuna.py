from sklearn.impute import SimpleImputer
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class SimpleImputerOptuna(SimpleImputer):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SimpleImputer_')

        self.strategy = trial.suggest_categorical(self.name + "strategy", ["mean", "median", "most_frequent"])
