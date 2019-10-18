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



class HyperOptSelection(BaseEstimator, SelectorMixin):
	def __init__(self, selection_strategy, max_complexity=None, min_accuracy=None, model=None, parameters=None, cv=None, scoring=None):
		self.selection_strategy = selection_strategy
		self.my_pipeline = Pipeline([('select', self.selection_strategy),
									 ('model', model)
									 ])
		self.parameters = parameters
		self.cv = cv
		self.max_complexity = max_complexity
		self.min_accuracy = min_accuracy
		self.scoring = scoring



	def fit(self, X, y=None):
		map_k_to_result = {}


		def objective(k):
			if k in map_k_to_result:
				return map_k_to_result[k]
			print(k)
			parameters = copy.deepcopy(self.parameters)
			parameters['select__' + 'k'] = [int(k)]

			cv_eval = GridSearchCV(estimator=self.my_pipeline, param_grid=parameters, cv=self.cv, scoring=self.scoring)
			cv_eval.fit(X, y)

			result = {'loss': -1 * cv_eval.best_score_, 'status': STATUS_OK, 'mask': cv_eval.best_estimator_.named_steps['select']._get_support_mask()}
			map_k_to_result[k] = result
			return result

		trials = Trials()
		best = None
		space = None
		for i in range(1, self.max_complexity):
			if i == 1:
				space = scope.int(hp.quniform('k', self.max_complexity, self.max_complexity, 1))
			else:
				space = scope.int(hp.quniform('k', 1, self.max_complexity, 1))

			fmin(objective, space=space, algo=tpe.suggest, max_evals=i, trials=trials)
			if trials.best_trial['result']['loss'] * -1 >= self.min_accuracy:
				self.mask_ = trials.best_trial['result']['mask']
				return self



	def _get_support_mask(self):
		return self.mask_