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
import time
from sklearn.model_selection import StratifiedKFold
import pickle

class ForwardSequentialSelection(BaseEstimator, SelectorMixin):
	def __init__(self, max_complexity=None, min_accuracy=None, model=None, parameters=None, kfold=None, scoring=None, step_size=1, fit_time_out=None, X_test=None, y_test=None):
		self.parameters = parameters
		self.kfold = kfold
		self.max_complexity = max_complexity
		self.min_accuracy = min_accuracy
		self.scoring = scoring
		self.step_size = step_size
		self.model = model
		self.fit_time_out = fit_time_out
		self.X_test = X_test
		self.y_test = y_test

	def fit(self, X, y=None):
		start_time = time.time()

		print('see shape: ' + str(X.shape[1]))

		rest_features = list(range(X.shape[1]))
		current_best_features = []

		self.map_k_to_results = {}

		for k in range(1, self.max_complexity + 1):
			evaluations = []
			for i in range(len(rest_features)):
				print(i)
				current_feature_list = copy.deepcopy(current_best_features)
				current_feature_list.append(rest_features[i])

				#TODO: implement it the same way as RFE

				cv_eval = GridSearchCV(estimator=self.model, param_grid=self.parameters, cv=self.kfold, scoring=self.scoring)
				cv_eval.fit(X[:, current_feature_list], y)
				evaluations.append(cv_eval.best_score_)

				if k <= self.max_complexity and np.max(evaluations) >= self.min_accuracy:
					current_best_features.append(rest_features[np.argmax(evaluations)])
					self.mask_ = np.zeros(X.shape[1], dtype=bool)
					self.mask_[current_best_features] = True
					return self

				if type(self.fit_time_out) != type(None) and self.fit_time_out < time.time() - start_time:
					return self

			best_feature_id = np.argmax(evaluations)
			best_cv_score = max(evaluations)
			current_best_features.append(rest_features[best_feature_id])

			test_score = cv_eval.score(self.X_test[:, current_best_features], self.y_test)
			self.map_k_to_results[k] = (cv_eval.best_score_, time.time() - start_time, test_score)

			pfile = open("/tmp/allforward.p", "wb")
			pickle.dump(self.map_k_to_results, pfile)
			pfile.flush()
			pfile.close()


			del rest_features[best_feature_id]

	def _get_support_mask(self):
		return self.mask_