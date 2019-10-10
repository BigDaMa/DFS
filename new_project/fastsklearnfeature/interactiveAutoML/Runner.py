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
import numpy as np
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
import pandas as pd
import time

class Runner:
	def __init__(self, c=None, sensitive=None, labels = None, experiment='experiment3'):
		self.which_experiment = experiment
		self.global_starting_time = time.time()

		self.numeric_representations = pickle.load(
			open("/home/felix/phd/feature_constraints/" + str(self.which_experiment) + "/features.p", "rb"))

		# print(len(self.numeric_representations))

		# X_train, X_test, y_train, y_test
		self.X_train = pickle.load(
			open("/home/felix/phd/feature_constraints/" + str(self.which_experiment) + "/X_train.p", "rb"))
		self.X_test = pickle.load(
			open("/home/felix/phd/feature_constraints/" + str(self.which_experiment) + "/X_test.p", "rb"))
		self.y_train = pickle.load(
			open("/home/felix/phd/feature_constraints/" + str(self.which_experiment) + "/y_train.p", "rb")).values
		self.y_test = pickle.load(
			open("/home/felix/phd/feature_constraints/" + str(self.which_experiment) + "/y_test.p", "rb"))

		self.model = LogisticRegression

		self.c = c

		if type(self.c) == type(None):
			self.c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
		else:
			self.c = [c]

		self.parameter_grid = {'c__penalty': ['l2'], 'c__C': self.c, 'c__solver': ['lbfgs'],
						  'c__class_weight': ['balanced'], 'c__max_iter': [10000], 'c__multi_class': ['auto']}

		self.scoring = {'auc': make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)}
		if type(sensitive) != type(None):
			self.scoring['fair'] = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=sensitive, labels=labels)



	def run_pipeline(self, which_features_to_use, runs=1):
		results = {}

		start_time = time.time()

		# generate pipeline
		results['complexity']=0
		all_selected_features = []
		for i in range(len(which_features_to_use)):
			if which_features_to_use[i]:
				all_selected_features.append(self.numeric_representations[i])
				results['complexity'] += self.numeric_representations[i].get_complexity()

		all_features = CandidateFeature(IdentityTransformation(-1), all_selected_features)
		all_standardized = CandidateFeature(MinMaxScalingTransformation(), [all_features])

		my_pipeline = Pipeline([('f', all_standardized.pipeline),
								('c', self.model())
								])

		cv_scores = []
		test_scores = []
		pred_test = None
		proba_pred_test = None

		if runs > 1:
			for r in range(runs):
				kfolds = StratifiedKFold(10, shuffle=True, random_state=42+r)
				self.pipeline = GridSearchCV(my_pipeline, self.parameter_grid, cv=kfolds.split(self.X_train, self.y_train), scoring=self.scoring, n_jobs=4)
				self.pipeline.fit(self.X_train, self.y_train)

				pred_test = self.pipeline.predict(self.X_test)
				proba_pred_test = self.pipeline.predict_proba(self.X_test)

				test_auc = self.auc(self.pipeline, self.X_test, self.y_test)

				cv_scores.append(self.pipeline.best_score_)
				test_scores.append(test_auc)

			std_loss = np.std(cv_scores)
			loss = np.average(cv_scores)
		else:
			kfolds = StratifiedKFold(10, shuffle=True, random_state=42)
			self.pipeline = GridSearchCV(my_pipeline, self.parameter_grid, cv=kfolds.split(self.X_train, self.y_train), scoring=self.scoring, n_jobs=1, refit='auc')
			self.pipeline.fit(self.X_train, pd.DataFrame(self.y_train))

			pred_test = self.pipeline.predict(self.X_test)
			proba_pred_test = self.pipeline.predict_proba(self.X_test)

			test_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)(self.pipeline, self.X_test, self.y_test)

			for k in self.scoring.keys():
				results[k] = self.pipeline.cv_results_['mean_test_'+str(k)][self.pipeline.best_index_]


			loss = self.pipeline.cv_results_['mean_test_auc'][self.pipeline.best_index_]
			test_scores.append(test_auc)

		results['test_auc'] = np.average(test_scores)

		results['cv_time'] = time.time() - start_time
		results['global_time'] = time.time() - self.global_starting_time

		return results#loss, np.average(test_scores), pred_test, 0.0, proba_pred_test





