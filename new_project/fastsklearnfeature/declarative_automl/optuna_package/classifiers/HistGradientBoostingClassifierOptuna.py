from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

class HistGradientBoostingClassifierOptuna(HistGradientBoostingClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'HistGradientBoostingClassifier_'
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

