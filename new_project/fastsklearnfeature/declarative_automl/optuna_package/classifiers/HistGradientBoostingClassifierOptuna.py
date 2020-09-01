from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class HistGradientBoostingClassifierOptuna(HistGradientBoostingClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('HistGradientBoostingClassifier_')
        self.loss = "auto"
        self.learning_rate = trial.suggest_loguniform(self.name + "learning_rate", 0.01, 1)
        self.min_samples_leaf = trial.suggest_int(self.name + "min_samples_leaf", 1, 200, log=True)
        self.max_depth = None
        self.max_leaf_nodes = trial.suggest_int(self.name + "max_leaf_nodes", 3, 2047, log=True)
        self.max_bins = 255
        self.l2_regularization = trial.suggest_loguniform(self.name + "l2_regularization", 1E-10, 1)
        self.early_stop = trial.suggest_categorical(self.name + "early_stop", ["off", "train", "valid"])
        self.tol = 1e-7
        self.scoring = "loss"
        self.vn_iter_no_change = trial.suggest_int(self.name + "n_iter_no_change", 1, 20)
        self.validation_fraction = trial.suggest_uniform(self.name + "validation_fraction", 0.01, 0.4)
        self.classes_ = np.unique(y.astype(int))

        #new
        self.max_iter = trial.suggest_int(self.name + "max_iter", 10, 512, log=True)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('HistGradientBoostingClassifier_')

        space_gen.generate_number(self.name + "learning_rate", 0.1, depending_node=depending_node)
        space_gen.generate_number(self.name + "min_samples_leaf", 20, depending_node=depending_node)
        space_gen.generate_number(self.name + "max_leaf_nodes", 31, depending_node=depending_node)
        space_gen.generate_number(self.name + "l2_regularization", 1E-10, depending_node=depending_node)
        space_gen.generate_cat(self.name + "early_stop", ["off", "train", "valid"], "off", depending_node=depending_node)
        space_gen.generate_number(self.name + "n_iter_no_change", 10, depending_node=depending_node)
        space_gen.generate_number(self.name + "validation_fraction", 0.1, depending_node=depending_node)

        space_gen.generate_number(self.name + "max_iter", 100, depending_node=depending_node)


