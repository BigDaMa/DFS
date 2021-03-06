import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time


from fastsklearnfeature.interactiveAutoML.feature_selection.HyperOptSelection import HyperOptSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.ALSelectionK import ALSelectionK
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from skrebate import ReliefF


'''
#######################################################################################################
Recursive Feature Elimination
#######################################################################################################
'''

def run_sequential_search(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):

	start_time = time.time()
	inner_pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])

	if one_hot:
		my_pipeline = Pipeline([('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)),('selection', BackwardSelection(SelectKBest(score_func=mutual_info_classif), max_complexity=max_complexity, min_accuracy=min_accuracy, model=inner_pipeline, parameters={}, kfold=kfold, scoring=scoring, fit_time_out=fit_time_out)),
								('cmodel', model)
								])
	else:
		my_pipeline = Pipeline([('selection',
																									BackwardSelection(
																										SelectKBest(
																											score_func=mutual_info_classif),
																										max_complexity=max_complexity,
																										min_accuracy=min_accuracy,
																										model=inner_pipeline,
																										parameters={},
																										kfold=kfold,
																										scoring=scoring,
																										fit_time_out=fit_time_out)),
								('cmodel', model)
								])


	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time


'''
#######################################################################################################
KBest Strategies:
#######################################################################################################
'''

def run_kbest(score_function, X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):

	start_time = time.time()
	inner_pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])

	if one_hot:
		my_pipeline = Pipeline([('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)),('selection', HyperOptSelection(SelectKBest(score_func=score_function), max_complexity=max_complexity, min_accuracy=min_accuracy, model=inner_pipeline, parameters={}, cv=kfold, scoring=scoring, fit_time_out=fit_time_out)),
								('cmodel', model)
								])
	else:
		my_pipeline = Pipeline([('selection',
																									HyperOptSelection(
																										SelectKBest(
																											score_func=score_function),
																										max_complexity=max_complexity,
																										min_accuracy=min_accuracy,
																										model=inner_pipeline,
																										parameters={},
																										cv=kfold,
																										scoring=scoring,
																										fit_time_out=fit_time_out)),
								('cmodel', model)
								])

	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time

#A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual information". Phys. Rev. E 69, 2004.
def run_hyperopt_search_kbest_info(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(mutual_info_classif, X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)

def run_hyperopt_search_kbest_f_classif(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(f_classif, X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)


#Liu, H. and Setiono, R., 1995, November. Chi2: Feature selection and discretization of numeric attributes. In Proceedings of 7th IEEE International Conference on Tools with Artificial Intelligence (pp. 388-391). IEEE.
def run_hyperopt_search_kbest_chi2(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(chi2, X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)

#Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R.P., Tang, J. and Liu, H., 2018. Feature selection: A data perspective. ACM Computing Surveys (CSUR), 50(6), p.94.
def run_hyperopt_search_kbest_variance(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(variance, X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)

#Yu, L. and Liu, H., 2003. Feature selection for high-dimensional data: A fast correlation-based filter solution. In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 856-863).
def run_hyperopt_search_kbest_fcbf(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(fcbf, X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)



'''
#######################################################################################################
KBest Strategies based on models:
#######################################################################################################
'''

def bindFunction1(estimator):
    def func1(X,y):
        return model_score(X, y, estimator=estimator)
    func1.__name__ = 'score_model_' + estimator.__class__.__name__
    return func1

def run_hyperopt_search_kbest_forest(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	estimator = ExtraTreesClassifier(n_estimators=1000, random_state=0)
	return run_kbest(bindFunction1(estimator), X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)

def run_hyperopt_search_kbest_l1(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	estimator = LinearSVC(penalty="l1", dual=False)
	return run_kbest(bindFunction1(estimator), X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)


def run_hyperopt_search_kbest_relieff(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	estimator = ReliefF(n_neighbors=10)
	return run_kbest(bindFunction1(estimator), X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)



'''
#######################################################################################################
Forward selection
#######################################################################################################
'''

def run_forward_seq_search(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):

	start_time = time.time()
	inner_pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])

	if one_hot:
		my_pipeline = Pipeline([('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)),('selection', ForwardSequentialSelection(max_complexity=max_complexity, min_accuracy=min_accuracy, model=inner_pipeline, parameters={}, kfold=kfold, scoring=scoring, fit_time_out=fit_time_out)),
								('cmodel', model)
								])
	else:
		my_pipeline = Pipeline([('selection',
										ForwardSequentialSelection(
											max_complexity=max_complexity,
											min_accuracy=min_accuracy,
											model=inner_pipeline,
											parameters={},
											kfold=kfold,
											scoring=scoring,
											fit_time_out=fit_time_out)),
								('cmodel', model)
								])

	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time


'''
#######################################################################################################
Kth Active learning
#######################################################################################################
'''

def run_al_k_search(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):

	start_time = time.time()

	inner_pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])

	if one_hot:
		my_pipeline = Pipeline([('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)),('selection', ALSelectionK(max_complexity=max_complexity, min_accuracy=min_accuracy, model=inner_pipeline, parameters={}, kfold=kfold, scoring=scoring, fit_time_out=fit_time_out)),
								('cmodel', model)
								])
	else:
		my_pipeline = Pipeline([('selection',
																									ALSelectionK(
																										max_complexity=max_complexity,
																										min_accuracy=min_accuracy,
																										model=inner_pipeline,
																										parameters={},
																										kfold=kfold,
																										scoring=scoring,
																										fit_time_out=fit_time_out)),
								('cmodel', model)
								])

	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time

def my_function(id):
	X_train = my_global_utils1.X_train
	y_train = my_global_utils1.y_train
	model = my_global_utils1.model
	kfold = my_global_utils1.kfold
	scoring = my_global_utils1.scoring
	forward = my_global_utils1.forward
	max_complexity = my_global_utils1.max_complexity
	min_accuracy = my_global_utils1.min_accuracy
	fit_time_out = my_global_utils1.fit_time_out

	start = time.time()

	try:
		run_sequential_search(X_train, y_train, model=model, kfold=kfold,
							  scoring=scoring,
							  forward=forward,
							  max_complexity=max_complexity,
							  min_accuracy=min_accuracy,
							  fit_time_out=fit_time_out)
	except:
		pass

	return time.time() - start

