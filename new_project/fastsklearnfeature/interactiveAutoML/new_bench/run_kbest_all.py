import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
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
from sklearn.neighbors import KNeighborsClassifier
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


from fastsklearnfeature.interactiveAutoML.feature_selection.RunAllKBestSelection import RunAllKBestSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier

from fastsklearnfeature.configuration.Config import Config

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score

def run_kbest(score_function, X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):

	start_time = time.time()
	inner_pipeline = Pipeline([('scale', MinMaxScaler()), ('model', model)])

	if one_hot:
		inner_pipeline = Pipeline([('scale', MinMaxScaler()), ('model', model)])
		my_pipeline = Pipeline([('selection', RunAllKBestSelection(SelectKBest(score_func=score_function), max_complexity=max_complexity, min_accuracy=min_accuracy, model=inner_pipeline, parameters={}, cv=kfold, scoring=scoring, fit_time_out=fit_time_out,X_test=X_test,y_test=y_test)),
								('cmodel', model)
								])
	else:
		my_pipeline = Pipeline([('selection',
										RunAllKBestSelection(
											SelectKBest(
												score_func=score_function),
											max_complexity=max_complexity,
											min_accuracy=min_accuracy,
											model=inner_pipeline,
											parameters={},
											cv=kfold,
											scoring=scoring,
											fit_time_out=fit_time_out,
											X_test=X_test,
											y_test=y_test)
								 ),
								('cmodel', model)
								])

	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time

#A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual information". Phys. Rev. E 69, 2004.
def run_hyperopt_search_kbest_info(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(mutual_info_classif, X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)

def run_hyperopt_search_kbest_f_classif(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(f_classif, X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)


#Liu, H. and Setiono, R., 1995, November. Chi2: Feature selection and discretization of numeric attributes. In Proceedings of 7th IEEE International Conference on Tools with Artificial Intelligence (pp. 388-391). IEEE.
def run_hyperopt_search_kbest_chi2(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(chi2, X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)

#Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R.P., Tang, J. and Liu, H., 2018. Feature selection: A data perspective. ACM Computing Surveys (CSUR), 50(6), p.94.
def run_hyperopt_search_kbest_variance(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(variance, X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy, fit_time_out, one_hot)

#Yu, L. and Liu, H., 2003. Feature selection for high-dimensional data: A fast correlation-based filter solution. In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 856-863).
def run_hyperopt_search_kbest_fcbf(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(fcbf, X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)


'''
#######################################################################################################
new:
#######################################################################################################
'''
def run_hyperopt_search_kbest_fisher_score(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	return run_kbest(my_fisher_score, X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy,
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

def run_hyperopt_search_kbest_forest(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	estimator = ExtraTreesClassifier(n_estimators=1000, random_state=0)
	return run_kbest(bindFunction1(estimator), X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)

def run_hyperopt_search_kbest_l1(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	estimator = LinearSVC(penalty="l1", dual=False)
	return run_kbest(bindFunction1(estimator), X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)


def run_hyperopt_search_kbest_relieff(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):
	estimator = ReliefF(n_neighbors=10)
	return run_kbest(bindFunction1(estimator), X_train, y_train, X_test, y_test, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out, one_hot)




def run_rfe_search(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):

	start_time = time.time()
	inner_pipeline = Pipeline([('scale', MinMaxScaler()), ('model', model)])

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
											fit_time_out=fit_time_out,X_test=X_test,y_test=y_test)),
							('cmodel', model)
							])


	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time


def run_forward_search(X_train, y_train, X_test=None, y_test=None, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None, one_hot=False):

	start_time = time.time()
	inner_pipeline = Pipeline([('scale', MinMaxScaler()), ('model', model)])

	my_pipeline = Pipeline([('selection',
							 ForwardSequentialSelection(
								 max_complexity=max_complexity,
								 min_accuracy=min_accuracy,
								 model=inner_pipeline,
								 parameters={},
								 kfold=kfold,
								 scoring=scoring,
								 fit_time_out=fit_time_out,X_test=X_test,y_test=y_test)),
							('cmodel', model)
							])


	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time


'''
data = pd.read_csv(Config.get('data_path') + '/breastTumor/breastTumor.csv', delimiter=',', header=0)
y = data['binaryClass'].values
X = data[data.columns.difference(['binaryClass'])].values
data_name = 'breastTumor'
one_hot = True
'''

'''
data = pd.read_csv(Config.get('data_path') + '/promoters/dataset_106_molecular-biology_promoters.csv', delimiter=',', header=0)
y = data['class'].values
X = data[data.columns.difference(['class', 'instance'])].values
data_name = 'promoters'
one_hot = True
'''





X = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.data', delimiter=' ', header=None).values[:,0:500]
y = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.labels', delimiter=' ', header=None).values
data_name = 'madelon'
one_hot = False


'''
X_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000]
y_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.labels', delimiter=' ', header=None).values
data_name = 'ARCENE'
one_hot = False
'''

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

xshape = X_train.shape[1]
if one_hot:
	encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
	X_train = encoder.fit_transform(X_train)
	xshape = X_train.shape[1]
	print(xshape)
	X_test = encoder.transform(X_test)


#X_train = X_train[0:100,:]
#y_train = y_train[0:100]


kfold = StratifiedKFold(n_splits=10, shuffle=False)
scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

'''
my_search_strategies = [run_hyperopt_search_kbest_forest, run_hyperopt_search_kbest_l1,
									   run_hyperopt_search_kbest_fcbf, run_hyperopt_search_kbest_relieff,
									   run_hyperopt_search_kbest_info, run_hyperopt_search_kbest_chi2, run_hyperopt_search_kbest_f_classif, run_hyperopt_search_kbest_variance,
									  ]
'''
my_search_strategies = [run_hyperopt_search_kbest_forest, run_hyperopt_search_kbest_l1,
									   run_hyperopt_search_kbest_fcbf, run_hyperopt_search_kbest_relieff,
									   run_hyperopt_search_kbest_info, run_hyperopt_search_kbest_chi2, run_hyperopt_search_kbest_f_classif, run_hyperopt_search_kbest_variance,
						run_rfe_search, run_forward_search]


my_search_strategies = [run_hyperopt_search_kbest_l1]

print(xshape)

for strategy in my_search_strategies:
	#try:
	runtime = strategy (X_train, y_train, X_test, y_test,
							#model=DecisionTreeClassifier(),
							#model=LogisticRegression(),
							model=KNeighborsClassifier(n_neighbors=3),
							kfold=copy.deepcopy(kfold),
							scoring=scoring,
							max_complexity=int(xshape),
							min_accuracy=np.inf,
							fit_time_out=np.inf,
							one_hot=False
						 )
	#except:
	#	pass