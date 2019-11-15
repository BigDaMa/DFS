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

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier


from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score

'''
#######################################################################################################
KBest Strategies:
#######################################################################################################
'''

def bindFunction1(estimator):
    def func1(X,y):
        return model_score(X, y, estimator=estimator)
    func1.__name__ = 'score_model_' + estimator.__class__.__name__
    return func1

def run_kbest(score_function, X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None):

	start_time = time.time()
	inner_pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])
	my_pipeline = Pipeline([('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)),('selection', HyperOptSelection(SelectKBest(score_func=score_function), max_complexity=max_complexity, min_accuracy=min_accuracy, model=inner_pipeline, parameters={}, cv=kfold, scoring=scoring, fit_time_out=fit_time_out)),
							('cmodel', model)
							])

	parameter_grid = {}
	my_pipeline.set_params(**parameter_grid)
	my_pipeline.fit(X_train, pd.DataFrame(y_train))
	return time.time() - start_time

def run_hyperopt_search_kbest_forest(X_train, y_train, model=None, kfold=None, scoring=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), max_complexity=None, min_accuracy=None, fit_time_out=None):
	estimator = ExtraTreesClassifier(n_estimators=1000, random_state=0)
	return run_kbest(bindFunction1(estimator), X_train, y_train, model, kfold, scoring, max_complexity, min_accuracy,
					 fit_time_out)
