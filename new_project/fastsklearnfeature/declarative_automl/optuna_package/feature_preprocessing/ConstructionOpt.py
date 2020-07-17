from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class ConstructionOpt(ConstructionTransformer):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('ConstructionTransformer_')

        #t = ConstructionTransformer(c_max=3, epsilon=-np.inf, scoring=auc, n_jobs=4, model=LogisticRegression(), cv=2, feature_names=attribute_names, feature_is_categorical=categorical_indicator,parameter_grid={'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10], 'multi_class':['auto']})

        self.c_max = trial.suggest_int(self.name + "c_max", 2, 4)
        self.epsilon = -np.inf
        self.cv = 2
        self.model = LogisticRegression()
        self.parameter_grid = {'penalty': ['l2'], 'C': [1], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10], 'multi_class':['auto']}
        self.n_jobs = 1
