from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestClassifierOptuna(RandomForestClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = 'RandomForestClassifier_'
        self.criterion = trial.suggest_categorical(self.name + "criterion", ["gini", "entropy"])
        self.max_features = trial.suggest_uniform(self.name + "max_features", 0., 1.)
        self.max_depth = None
        self.min_samples_split = trial.suggest_int(self.name + "min_samples_split", 2, 20, log=False)
        self.min_samples_leaf = trial.suggest_int(self.name + "min_samples_leaf", 1, 20, log=False)
        self.min_weight_fraction_leaf = 0.
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.bootstrap = trial.suggest_categorical(self.name + "bootstrap", [True, False])
        self.classes_ = np.unique(y.astype(int))
