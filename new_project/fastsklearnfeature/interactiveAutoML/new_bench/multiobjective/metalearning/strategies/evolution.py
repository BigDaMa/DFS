from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
from pymoo.model.termination import Termination
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.cheating_global as cheating_global
import random
from pymoo.model.repair import Repair


def evolution(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy=0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, cv_splitter=None):

	def calculate_loss(cv_acc, cv_fair, cv_robust, cv_number_features):
		loss = 0.0
		if cv_acc >= min_accuracy and \
				cv_fair >= min_fairness and \
				cv_robust >= min_robustness:
			if min_fairness > 0.0:
				loss += (min_fairness - cv_fair)
			if min_accuracy > 0.0:
				loss += (min_accuracy - cv_acc)
			if min_robustness > 0.0:
				loss += (min_robustness - cv_robust)
		else:
			if min_fairness > 0.0 and cv_fair < min_fairness:
				loss += (min_fairness - cv_fair) ** 2
			if min_accuracy > 0.0 and cv_acc < min_accuracy:
				loss += (min_accuracy - cv_acc) ** 2
			if min_robustness > 0.0 and cv_robust < min_robustness:
				loss += (min_robustness - cv_robust) ** 2
		return loss

	hash = str(random.getrandbits(128)) + str(time.time())
	cheating_global.successfull_result[hash] = {}
	cheating_global.successfull_result[hash]['cv_number_evaluations'] = 0

	start_time = time.time()

	auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
	fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
	fair_test = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_test[:, sensitive_ids[0]])


	def f_clf1(mask):
		model = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', LogisticRegression())
		])
		return model

	# define an objective function
	def objective(features):
		if 'time' in cheating_global.successfull_result[hash]:
			return [0.0, 0.0, 0.0, 0.0]

		if np.sum(features) == 0:
			return [0.0, 0.0, 0.0, 0.0]

		model = f_clf1(features)

		robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train, y=y_train, model=clf,
									feature_selector=model.named_steps['selection'], scorer=auc_scorer)

		cv = GridSearchCV(model, param_grid={'clf__C': [1.0]}, cv=cv_splitter,
						  scoring={'AUC': auc_scorer, 'Fairness': fair_train, 'Robustness': robust_scorer},
						  refit=False)
		cv.fit(X_train, pd.DataFrame(y_train))
		cv_acc = cv.cv_results_['mean_test_AUC'][0]
		cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]
		cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

		cheating_global.successfull_result[hash]['cv_number_evaluations'] += 1

		cv_number_features = float(np.sum(model.named_steps['selection']._get_support_mask())) / float(
			len(model.named_steps['selection']._get_support_mask()))

		cv_simplicity = 1.0 - cv_number_features

		if not 'cv_acc' in cheating_global.successfull_result[hash] or\
				calculate_loss(cheating_global.successfull_result[hash]['cv_acc'],
					  cheating_global.successfull_result[hash]['cv_fair'],
					  cheating_global.successfull_result[hash]['cv_robust'],
					  cheating_global.successfull_result[hash]['cv_number_features']
					  ) > calculate_loss(cv_acc, cv_fair, cv_robust, cv_number_features):
			cheating_global.successfull_result[hash]['cv_acc'] = cv_acc
			cheating_global.successfull_result[hash]['cv_robust'] = cv_robust
			cheating_global.successfull_result[hash]['cv_fair'] = cv_fair
			cheating_global.successfull_result[hash]['cv_number_features'] = cv_number_features


		#check constraints for test set
		if cv_fair >= min_fairness and cv_acc >= min_accuracy and cv_robust >= min_robustness and cv_number_features <= max_number_features:
			model.fit(X_train, pd.DataFrame(y_train))

			test_acc = 0.0
			if min_accuracy > 0.0:
				test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
			test_fair = 0.0
			if min_fairness > 0.0:
				test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
			test_robust = 0.0
			if min_robustness > 0.0:
				test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test, model=model.named_steps['clf'], feature_selector=model.named_steps['selection'], scorer=auc_scorer)

			if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
				print('fair: ' + str(min(cv_fair, test_fair)) + ' acc: ' + str(min(cv_acc, test_acc)) + ' robust: ' + str(min(test_robust, cv_robust)) + ' k: ' + str(cv_number_features))
				cheating_global.successfull_result[hash]['time'] = time.time() - start_time
				cheating_global.successfull_result[hash]['cv_acc'] = cv_acc
				cheating_global.successfull_result[hash]['cv_robust'] = cv_robust
				cheating_global.successfull_result[hash]['cv_fair'] = cv_fair
				cheating_global.successfull_result[hash]['cv_number_features'] = cv_number_features

		return [cv_acc, cv_fair, cv_robust, cv_simplicity]
	
	class MyProblem(Problem):
	
		def __init__(self):
			number_objectives = 0
			if min_accuracy > 0.0:
				number_objectives += 1
			if min_fairness > 0.0:
				number_objectives += 1
			if min_robustness > 0.0:
				number_objectives += 1
			if number_objectives == 0:
				number_objectives = 3

			super().__init__(n_var=X_train.shape[1],
								 n_obj=number_objectives,
								 n_constr=0,
								 xl=0, xu=1, type_var=anp.bool)
	
		def _evaluate(self, x, out, *args, **kwargs):
			accuracy_batch = []
			fairness_batch = []
			robustness_batch = []
			simplicity_batch = []
	
			for i in range(len(x)):
				results = objective(x[i])
				accuracy_batch.append(results[0]*-1)#accuracy
				fairness_batch.append(results[1]*-1)#fairness
				robustness_batch.append(results[2]*-1)#robustness
				simplicity_batch.append(results[3]*-1)  # simplicity

			##objectives
			objectives = []
			if min_accuracy > 0.0 or self.n_obj == 3:
				objectives.append(accuracy_batch)
			if min_fairness > 0.0 or self.n_obj == 3:
				objectives.append(fairness_batch)
			if min_robustness > 0.0 or self.n_obj == 3:
				objectives.append(robustness_batch)
	
			out["F"] = anp.column_stack(objectives)
	

	problem = MyProblem()

	class NumberFeaturesRepair(Repair):

		def __init__(self, max_number_features=None):
			self.max_number_features = max_number_features

		def _do(self, problem, pop, **kwargs):
			# the packing plan for the whole population (each row one individual)
			Z = pop.get("X")
			# now repair each indvidiual i
			for i in range(len(Z)):
				if np.sum(Z[i]) > self.max_number_features:
					id_features_used = np.nonzero(Z[i])[0] #indices where features are used
					np.random.shuffle(id_features_used) #shuffle ids
					ids_tb_deactived = id_features_used[self.max_number_features:]# deactivate features
					for item_to_remove in ids_tb_deactived:
						Z[i][item_to_remove] = False

			# set the design variables for the population
			pop.set("X", Z)
			return pop

	repair_strategy = None
	if max_number_features < 1.0:
		max_k = max(int(max_number_features * X_train.shape[1]), 1)
		repair_strategy = NumberFeaturesRepair(max_k)
	
	population_size = 100
	cross_over_rate = 0.9
	algorithm = NSGA2(pop_size=population_size,
					  sampling=get_sampling("bin_random"),
					  crossover=get_crossover('bin_one_point'),#get_crossover("bin_hux"),#get_crossover("bin_two_point"),
					  mutation=BinaryBitflipMutation(1.0 / X_train.shape[1]),
					  elimate_duplicates=True,
					  repair=repair_strategy,
					  #n_offsprings= cross_over_rate * population_size
					  )
		
	
	class MyTermination(Termination):
	
		def __init__(self, start_time=None, time_limit=None) -> None:
			super().__init__()
			self.start_time = start_time
			self.time_limit = time_limit
	
		def _do_continue(self, algorithm):
			if 'time' in cheating_global.successfull_result[hash] or (time.time() - self.start_time) > self.time_limit:
				return False
	
			return True

	minimize(problem=problem, algorithm=algorithm, termination=MyTermination(start_time, max_search_time), disp=False)

	runtime = time.time() - start_time
	success = 'time' in cheating_global.successfull_result[hash]
	cv_acc = cheating_global.successfull_result[hash]['cv_acc']
	cv_robust = cheating_global.successfull_result[hash]['cv_robust']
	cv_fair = cheating_global.successfull_result[hash]['cv_fair']
	cv_number_features = cheating_global.successfull_result[hash]['cv_number_features']
	number_of_evaluations = cheating_global.successfull_result[hash]['cv_number_evaluations']
	del cheating_global.successfull_result[hash]
	return {'time': runtime, 'success': success, 'cv_acc': cv_acc, 'cv_robust': cv_robust, 'cv_fair': cv_fair, 'cv_number_features': cv_number_features, 'cv_number_evaluations': number_of_evaluations}
