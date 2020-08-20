from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
import sklearn
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

import functools

def model_score(X, y=None, estimator=None):
    estimator.fit(X,y)
    scores = _get_feature_importances(estimator)
    return scores

class SelectPercentileOptuna(SelectPercentile):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SelectPercentile')

        self.percentile = trial.suggest_int(self.name + "percentile", 1, 99)

        self.sparse = False



        score_func = trial.suggest_categorical(self.name + 'score_func', ['chi2', 'f_classif', 'mutual_info', 'ExtraTreesClassifier', 'LinearSVC'])

        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info":
            self.score_func = sklearn.feature_selection.mutual_info_classif

        elif score_func == 'ExtraTreesClassifier':
            new_name = self.name + '_' + score_func + '_'
            model = ExtraTreesClassifier()
            model.n_estimators = 100
            model.criterion = trial.suggest_categorical(new_name + "criterion", ["gini", "entropy"])
            model.max_features = trial.suggest_uniform(new_name + "max_features", 0, 1)
            model.max_depth = None
            model.max_leaf_nodes = None
            model.min_samples_split = trial.suggest_int(new_name + "min_samples_split", 2, 20, log=False)
            model.min_samples_leaf = trial.suggest_int(new_name + "min_samples_leaf", 1, 20, log=False)
            model.min_weight_fraction_leaf = 0.
            model.min_impurity_decrease = 0.
            model.bootstrap = trial.suggest_categorical(new_name + "bootstrap", [True, False])

            self.score_func = functools.partial(model_score, estimator=model) #bindFunction1(model)

        elif score_func == 'LinearSVC':
            new_name = self.name + '_' + score_func + '_'
            model = sklearn.svm.LinearSVC()
            model.penalty = "l1"
            model.loss = "squared_hinge"
            model.dual = False
            model.tol = trial.suggest_loguniform(new_name + "tol", 1e-5, 1e-1)
            model.C = trial.suggest_loguniform(new_name + "C", 0.03125, 32768)
            model.multi_class = "ovr"
            model.fit_intercept = True
            model.intercept_scaling = 1

            self.score_func = functools.partial(model_score, estimator=model)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SelectPercentile')

        space_gen.generate_number(self.name + "percentile", 50, depending_node=depending_node)
        category_fs = space_gen.generate_cat(self.name + 'score_func',['chi2', 'f_classif', 'mutual_info', 'ExtraTreesClassifier', 'LinearSVC',
                                                'variance'], "chi2", depending_node=depending_node)

        tree_catgory = category_fs[3]
        lr_catgory = category_fs[4]

        new_name = self.name + '_' + 'ExtraTreesClassifier' + '_'

        space_gen.generate_cat(new_name + "criterion", ["gini", "entropy"], "gini", depending_node=tree_catgory)
        space_gen.generate_number(new_name + "max_features", 0.5, depending_node=tree_catgory)
        space_gen.generate_number(new_name + "min_samples_split", 2, depending_node=tree_catgory)
        space_gen.generate_number(new_name + "min_samples_leaf", 1, depending_node=tree_catgory)
        space_gen.generate_cat(new_name + "bootstrap", [True, False], False, depending_node=tree_catgory)

        new_name = self.name + '_' + 'LinearSVC' + '_'
        space_gen.generate_cat(new_name + "loss", ["hinge", "squared_hinge"], "squared_hinge", depending_node=lr_catgory)
        space_gen.generate_number(new_name + "tol", 1e-4, depending_node=lr_catgory)
        space_gen.generate_number(new_name + "C", 1.0, depending_node=lr_catgory)











