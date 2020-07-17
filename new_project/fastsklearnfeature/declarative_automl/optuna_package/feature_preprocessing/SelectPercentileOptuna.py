from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
import sklearn
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

def model_score(X, y=None, estimator=None):
    estimator.fit(X,y)
    scores = _get_feature_importances(estimator)
    return scores

def bindFunction1(estimator):
    def func1(X,y):
        return model_score(X, y, estimator=estimator)
    func1.__name__ = 'score_model_' + estimator.__class__.__name__
    return func1

class SelectPercentileOptuna(SelectPercentile):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SelectPercentile')

        self.percentile = trial.suggest_int(self.name + "percentile", 1, 99)



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

            self.score_func = bindFunction1(model)

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

            self.score_func = bindFunction1(model)





