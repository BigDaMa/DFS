from sklearn.tree import DecisionTreeClassifier
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class DecisionTreeClassifierOptuna(DecisionTreeClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('DecisionTreeClassifier_')
        self.criterion = trial.suggest_categorical(self.name + "criterion", ["gini", "entropy"])
        self.max_depth_factor = trial.suggest_uniform(self.name + 'max_depth_factor', 0., 2.)
        self.min_samples_split = trial.suggest_int(self.name + "min_samples_split", 2, 20, log=False)
        self.min_samples_leaf = trial.suggest_int(self.name + "min_samples_leaf", 1, 20, log=False)
        min_weight_fraction_leaf = 0.0
        self.max_features = 1.0
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.classes_ = np.unique(y.astype(int))
