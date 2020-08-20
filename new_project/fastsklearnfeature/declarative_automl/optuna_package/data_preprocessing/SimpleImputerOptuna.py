from sklearn.impute import SimpleImputer
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np

class SimpleImputerOptuna(SimpleImputer):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SimpleImputer_')

        self.strategy = trial.suggest_categorical(self.name + "strategy", ["mean", "median", "most_frequent"])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SimpleImputer_')
        space_gen.generate_cat(self.name + "strategy", ["mean", "median", "most_frequent"], 'mean', depending_node=depending_node)
