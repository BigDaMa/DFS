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

class BackwardSelection(BaseEstimator, SelectorMixin):
	def __init__(self, selection_strategy, max_complexity=None, min_accuracy=None, model=None, parameters=None, kfold=None, scoring=None, step_size=1, fit_time_out=None):
		self.selection_strategy = selection_strategy
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

		self.log_results_ = []

		def update_ids(selector, ids):
			mask = selector._get_support_mask()

			assert len(ids) == len(mask)
			for d in range(len(mask)-1, -1, -1):
				if not mask[d]:
					del ids[d]
			return ids

		fold_ids = []
		for train, test in self.kfold.split(X, y):
			fold_ids.append((train, test))

		feature_ids_per_fold = []
		for fold_i in range(len(fold_ids)):
			feature_ids_per_fold.append(list(range(X.shape[1])))

		feature_ids_all = list(range(X.shape[1]))

		my_pipeline = Pipeline([
			('select', self.selection_strategy),
			('model', self.model)
		])

		for number_features in range(X.shape[1] - 1, 0, -1 * self.step_size):
			pipeline_per_fold = []
			score_per_fold = []

			parameters = {}
			parameters['select__' + 'k'] = number_features

			my_pipeline.set_params(**parameters)

			for fold_i in range(len(fold_ids)):
				my_pipeline.fit(X[fold_ids[fold_i][0]][:, feature_ids_per_fold[fold_i]], y.values[fold_ids[fold_i][0]])
				score_per_fold.append(self.scoring(my_pipeline, X[fold_ids[fold_i][1]][:, feature_ids_per_fold[fold_i]], y.values[fold_ids[fold_i][1]]))
				pipeline_per_fold.append(copy.deepcopy(my_pipeline))
				feature_ids_per_fold[fold_i] = update_ids(my_pipeline.named_steps['select'], feature_ids_per_fold[fold_i])

			print("cv: " + str(np.mean(score_per_fold)))
			my_pipeline.fit(X[:, feature_ids_all], y)
			feature_ids_all = update_ids(my_pipeline.named_steps['select'], feature_ids_all)

			self.log_results_.append([number_features, np.mean(score_per_fold), time.time() - start_time])

			if number_features <= self.max_complexity and np.mean(score_per_fold) >= self.min_accuracy:
				self.mask_ = np.zeros(X.shape[1], dtype=bool)
				self.mask_[feature_ids_all] = True
				return self

			if type(self.fit_time_out) != type(None) and self.fit_time_out < time.time() - start_time:
				return self

	def _get_support_mask(self):
		return self.mask_