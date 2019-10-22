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

class ForwardSequentialSelection(BaseEstimator, SelectorMixin):
	def __init__(self, max_complexity=None, min_accuracy=None, model=None, parameters=None, kfold=None, scoring=None, step_size=1, fit_time_out=None):
		self.parameters = parameters
		self.kfold = kfold
		self.max_complexity = max_complexity
		self.min_accuracy = min_accuracy
		self.scoring = scoring
		self.step_size = step_size
		self.model = model
		self.fit_time_out = fit_time_out

	def fit(self, X, y=None):
		start_time = time.time()

		rest_features = list(range(X.shape[1]))
		current_best_features = []

		for k in range(1, self.max_complexity + 1):
			evaluations = []
			for i in range(len(rest_features)):
				print(i)
				current_feature_list = rest_features
				current_feature_list.append(rest_features[i])

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

			best_feature_ids = np.argsort(np.array(evaluations) * -1)[0]
			current_best_features.append(rest_features[best_feature_ids])
			del rest_features[best_feature_ids]

	def _get_support_mask(self):
		return self.mask_