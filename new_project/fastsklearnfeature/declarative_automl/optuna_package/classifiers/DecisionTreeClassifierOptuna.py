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
        self.min_weight_fraction_leaf = 0.0
        self.max_features = 1.0
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.classes_ = np.unique(y.astype(int))

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):

        #set max depth
        max_depth_factor = None
        if type(self.max_depth_factor) != type(None):
            num_features = X.shape[1]
            self.max_depth_factor = int(self.max_depth_factor)
            max_depth_factor = max(1, int(np.round(self.max_depth_factor * num_features, 0)))
        self.max_depth = max_depth_factor

        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=X_idx_sorted)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('DecisionTreeClassifier_')
        space_gen.generate_cat(self.name + "criterion", ["gini", "entropy"], "gini", depending_node=depending_node)
        space_gen.generate_number(self.name + 'max_depth_factor', 0.5, depending_node=depending_node)
        space_gen.generate_number(self.name + "min_samples_split", 2, depending_node=depending_node)
        space_gen.generate_number(self.name + "min_samples_leaf", 1, depending_node=depending_node)
