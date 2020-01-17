from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sympy
import numpy as np
from sklearn.model_selection import GridSearchCV
import copy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import time
import pandas as pd
import inspect
import pickle
import multiprocessing as mp

import fastsklearnfeature.interactiveAutoML.feature_selection.WrapperBestK as wrap
from fastsklearnfeature.interactiveAutoML.feature_selection import my_global_utils_new

def k_cross_val(k):
	start_time = time.time()
	parameters = copy.deepcopy(my_global_utils_new.parameters)
	parameters['select__' + 'k'] = [int(k)]
	print(k)

	cv_eval = GridSearchCV(estimator=my_global_utils_new.my_pipeline, param_grid=parameters, cv=my_global_utils_new.cv, scoring=my_global_utils_new.scoring)
	cv_eval.fit(my_global_utils_new.X, pd.DataFrame(my_global_utils_new.y))

	#print(my_global_utils_new.X.shape)
	#print(my_global_utils_new.X_test.shape)

	test_score = cv_eval.score(my_global_utils_new.X_test, my_global_utils_new.y_test)

	#return test score too :)

	print(str({k: (cv_eval.best_score_, time.time() - start_time, test_score)}))

	return {k: (cv_eval.best_score_, time.time() - start_time, test_score)}


class RunAllKBestSelection(BaseEstimator, SelectorMixin):
	def __init__(self, selection_strategy, max_complexity=None, min_accuracy=None, model=None, parameters=None, cv=None, scoring=None, fit_time_out=None, X_test=None, y_test=None):
		self.selection_strategy = selection_strategy
		self.my_pipeline = Pipeline([('select', wrap.WrapperBestK(self.selection_strategy)),
									 ('model', model)
									 ])
		wrap.map_fold2ranking[self.selection_strategy.score_func.__name__] = {}
		print(self.selection_strategy.score_func.__name__)

		self.parameters = parameters
		self.cv = cv
		self.max_complexity = max_complexity
		self.min_accuracy = min_accuracy
		self.scoring = scoring
		self.fit_time_out = fit_time_out
		self.X_test = X_test
		self.y_test = y_test


	def fit(self, X, y=None):
		self.map_k_to_results = {}

		'''
		for k in range(1, self.max_complexity + 1):
			start_time = time.time()
			print(k)
			parameters = copy.deepcopy(self.parameters)
			parameters['select__' + 'k'] = [int(k)]

			cv_eval = GridSearchCV(estimator=self.my_pipeline, param_grid=parameters, cv=self.cv, scoring=self.scoring)
			cv_eval.fit(pd.DataFrame(X), y)

			self.map_k_to_results[k] = (cv_eval.best_score_, time.time() - start_time)
		'''

		my_global_utils_new.parameters = self.parameters
		my_global_utils_new.my_pipeline = self.my_pipeline
		my_global_utils_new.cv = self.cv
		my_global_utils_new.scoring = self.scoring
		my_global_utils_new.X = X
		my_global_utils_new.y = y

		my_global_utils_new.X_test = self.X_test
		my_global_utils_new.y_test = self.y_test

		n_jobs = mp.cpu_count()
		results = []

		results.append(k_cross_val(1))

		ranking_time = results[0][1]
		results.append({-1: ranking_time})

		with mp.Pool(processes=n_jobs) as pool:
			results.extend(pool.map(k_cross_val, range(1, self.max_complexity + 1)))

		for r in results:
			self.map_k_to_results.update(r)


		pfile = open("/tmp/all" + self.selection_strategy.score_func.__name__ + ".p", "wb")
		pickle.dump(self.map_k_to_results, pfile)
		pfile.flush()
		pfile.close()

		wrap.map_fold2ranking[self.selection_strategy.score_func.__name__] = {}

		return self



	def _get_support_mask(self):
		return self.mask_