from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.notebook.api.BaseSelection import BaseSelection
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import time
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_mcfs
from sklearn.model_selection import train_test_split
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from sklearn.feature_selection import mutual_info_classif
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from skrebate import ReliefF
import diffprivlib.models as models
from sklearn.model_selection import GridSearchCV

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.fullfeatures import fullfeatures
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import TPE
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import simulated_annealing
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.evolution import evolution
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.exhaustive import exhaustive
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.recursive_feature_elimination import recursive_feature_elimination

import matplotlib
import matplotlib.pyplot as plt

class QueryOptimizer(BaseSelection):
	def __init__(self):

		self.mappnames = {1: 'TPE(Variance)',
					 2: 'TPE($\chi^2$)',
					 3: 'TPE(FCBF)',
					 4: 'TPE(Fisher)',
					 5: 'TPE(MIM)',
					 6: 'TPE(MCFS)',
					 7: 'TPE(ReliefF)',
					 8: 'TPE(NR)',
					 9: 'SA(NR)',
					 10: 'NSGA-II(NR)',
					 11: 'ES(NR)',
					 12: 'SFS(NR)',
					 13: 'SBS(NR)',
					 14: 'SFFS(NR)',
					 15: 'SBFS(NR)',
					 16: 'RFE(LR)',
					 17: 'Complete Set'
					 }

		super(QueryOptimizer, self).__init__(None)


	def get_estimated_best_strategy(self, X_train, y_train,
									min_accuracy,
			  						sensitive_ids,
			 						min_fairness,
			  						min_safety,
			  						privacy,
			  						max_complexity,
			  						max_search_time):

		selection_strategies = {}
		rankings = {}

		ranking_list = [variance,
					chi2_score_wo,
					fcbf,
					my_fisher_score,
					mutual_info_classif,
					my_mcfs]
		ranking_list.append(partial(model_score, estimator=ReliefF(n_neighbors=10)))

		for my_strategy in range(1,8):
			selection_strategies[my_strategy] = weighted_ranking
			rankings[my_strategy] = ranking_list[my_strategy - 1]

		main_strategies = [TPE,
						   simulated_annealing,
						   evolution,
						   exhaustive,
						   forward_selection,
						   backward_selection,
						   forward_floating_selection,
						   backward_floating_selection,
						   recursive_feature_elimination,
						   fullfeatures]

		for my_strategy in range(8, 18):
			selection_strategies[my_strategy] = main_strategies[my_strategy - 8]
			rankings[my_strategy] = None




		if isinstance(max_complexity, int):
			max_complexity = max_complexity / float(X_train.shape[1])

		auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

		cv_splitter = StratifiedKFold(5, random_state=42)

		X_train_tiny, _, y_train_tiny, _ = train_test_split(X_train, y_train, train_size=100, random_state=42,
															stratify=y_train)

		fair_train_tiny = make_scorer(true_positive_rate_score, greater_is_better=True,
									  sensitive_data=X_train_tiny[:, sensitive_ids[0]])

		cv_k = 1.0
		model = LogisticRegression(class_weight='balanced')
		if type(privacy) == type(None):
			privacy = X_train_tiny.shape[0]
		else:
			model = models.LogisticRegression(epsilon=privacy, class_weight='balanced')

		robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train_tiny, y=y_train_tiny, model=model,
									feature_selector=None, scorer=auc_scorer)

		small_start_time = time.time()

		cv = GridSearchCV(model, param_grid={'C': [1.0]},
						  scoring={'AUC': auc_scorer, 'Fairness': fair_train_tiny, 'Robustness': robust_scorer},
						  refit=False, cv=cv_splitter)
		cv.fit(X_train_tiny, pd.DataFrame(y_train_tiny))
		cv_acc = cv.cv_results_['mean_test_AUC'][0]
		cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]
		cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

		cv_time = time.time() - small_start_time

		# construct feature vector
		feature_list = []
		# user-specified constraints
		feature_list.append(min_accuracy)
		feature_list.append(min_fairness)
		feature_list.append(max_complexity)
		feature_list.append(max_complexity * X_train.shape[1])
		feature_list.append(min_safety)
		feature_list.append(privacy)
		feature_list.append(max_search_time)
		# differences to sample performance
		feature_list.append(cv_acc - min_accuracy)
		feature_list.append(cv_fair - min_fairness)
		feature_list.append(cv_k - max_complexity)
		feature_list.append((cv_k - max_complexity) * X_train.shape[1])
		feature_list.append(cv_robust - min_safety)
		feature_list.append(cv_time)
		# privacy constraint is always satisfied => difference always zero => constant => unnecessary

		# metadata features
		feature_list.append(X_train.shape[0])  # number rows
		feature_list.append(X_train.shape[1])  # number columns

		features = np.array(feature_list).reshape(1, -1)

		self.predicted_probabilities = np.zeros(len(self.mappnames))

		for my_strategy in range(len(self.mappnames)):
			model = pickle.load(open('/tmp/model_strategy' + str(my_strategy) + '.pickle', "rb"))
			self.predicted_probabilities[my_strategy] = model.predict_proba(features)[:, 1]

		best_id = np.argmax(self.predicted_probabilities)

		self.selection_function = selection_strategies[best_id+1]
		self.ranking_functions = rankings[best_id+1]

	def get_plan(self,
			  X_train,
			  X_validation,
			  X_test,
			  y_train,
			  y_validation,
			  y_test,
			  classifier=LogisticRegression(class_weight='balanced'),
			  min_accuracy=0.5,
			  sensitive_ids=None,
			  min_fairness=0.0,
			  min_safety=0.0,
			  min_privacy=None,
			  max_complexity=1.0,
			  max_search_time=np.inf,
			  feature_names=None
			  ):
		self.get_estimated_best_strategy(X_train, y_train,
										 min_accuracy,
										 sensitive_ids,
										 min_fairness,
										 min_safety,
										 min_privacy,
										 max_complexity,
										 max_search_time)

		labels = []
		for my_strategy in range(len(self.mappnames)):
			labels.append(self.mappnames[my_strategy + 1])
		probas = self.predicted_probabilities

		plt.rcdefaults()
		fig, ax = plt.subplots()

		ids = np.argsort(probas)

		# Example data
		y_pos = np.arange(len(labels))
		ax.barh(y_pos, probas[ids], align='center')
		ax.set_yticks(y_pos)
		ax.set_yticklabels(np.array(labels)[ids])
		ax.invert_yaxis()  # labels read top-to-bottom
		ax.set_xlabel('Probabilities')
		ax.set_title('Strategies')
		plt.show()


	def query(self,
			  X_train,
			  X_validation,
			  X_test,
			  y_train,
			  y_validation,
			  y_test,
			  classifierm=LogisticRegression(class_weight='balanced'),
			  min_accuracy=0.5,
			  sensitive_ids=None,
			  min_fairness=0.0,
			  min_safety=0.0,
			  min_privacy=None,
			  max_complexity=1.0,
			  max_search_time=np.inf,
			  feature_names=None
			  ):
		self.get_estimated_best_strategy( X_train, y_train,
									min_accuracy,
									sensitive_ids,
									min_fairness,
									min_safety,
									min_privacy,
									max_complexity,
									max_search_time)
		return super().query(self,
			  X_train,
			  X_validation,
			  X_test,
			  y_train,
			  y_validation,
			  y_test,
			  classifier=classifierm,
			  min_accuracy=min_accuracy,
			  sensitive_ids=sensitive_ids,
			  min_fairness=min_fairness,
			  min_safety=min_safety,
			  min_privacy=min_privacy,
			  max_complexity=max_complexity,
			  max_search_time=max_search_time,
			  feature_names=feature_names
			  )