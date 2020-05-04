from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
from pymoo.model.termination import Termination
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.cheating_global as cheating_global
import random
from pymoo.model.repair import Repair
import pickle
import copy

def evolution(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy=0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None):

	hash = str(random.getrandbits(128)) + str(time.time())
	cheating_global.successfull_result[hash] = {}
	cheating_global.successfull_result[hash]['cv_number_evaluations'] = 0

	f_log = open(log_file, 'wb+')
	start_time = time.time()

	auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

	fair_validation = None
	fair_test = None
	if type(sensitive_ids) != type(None):
		fair_validation = make_scorer(true_positive_rate_score, greater_is_better=True,
									  sensitive_data=X_validation[:, sensitive_ids[0]])
		fair_test = make_scorer(true_positive_rate_score, greater_is_better=True,
								sensitive_data=X_test[:, sensitive_ids[0]])

	def f_clf1(mask):
		model = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', clf)
		])
		return model

	# define an objective function
	def objective(features, min_loss):
		if 'success' in cheating_global.successfull_result[hash]:
			return [0.0, 0.0, 0.0, 0.0, min_loss]

		if np.sum(features) == 0:
			return [0.0, 0.0, 0.0, 0.0, min_loss]

		pipeline = f_clf1(features)
		pipeline.fit(X_train, pd.DataFrame(y_train))

		cheating_global.successfull_result[hash]['cv_number_evaluations'] += 1

		validation_number_features = float(np.sum(pipeline.named_steps['selection']._get_support_mask())) / float(
			X_train.shape[1])

		validation_acc = auc_scorer(pipeline, X_validation, pd.DataFrame(y_validation))

		validation_fair = 0.0
		if type(sensitive_ids) != type(None) and min_fairness > 0.0:
			validation_fair = 1.0 - fair_validation(pipeline, X_validation, pd.DataFrame(y_validation))
		validation_robust = 0.0
		if min_robustness > 0.0:
			validation_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_validation, y_test=y_validation,
														model=pipeline.named_steps['clf'],
														feature_selector=pipeline.named_steps['selection'],
														scorer=auc_scorer)

		loss = 0.0
		if min_fairness > 0.0 and validation_fair < min_fairness:
			loss += (min_fairness - validation_fair) ** 2
		if min_accuracy > 0.0 and validation_acc < min_accuracy:
			loss += (min_accuracy - validation_acc) ** 2
		if min_robustness > 0.0 and validation_robust < min_robustness:
			loss += (min_robustness - validation_robust) ** 2
		print('evo: ' + str(loss))

		validation_simplicity = 1.0 - validation_number_features

		#check constraints for test set
		current_time = time.time() - start_time
		my_result = {'loss': loss,
					 'model': pipeline,
					 'cv_fair': validation_fair,
					 'cv_acc': validation_acc,
					 'cv_robust': validation_robust,
					 'cv_number_features': validation_number_features,
					 'time': current_time}

		my_result['number_evaluations'] = cheating_global.successfull_result[hash]['cv_number_evaluations']


		pipeline.fit(X_train_val, pd.DataFrame(y_train_val))

		test_acc = 0.0
		if min_accuracy > 0.0:
			test_acc = auc_scorer(pipeline, X_test, pd.DataFrame(y_test))
		test_fair = 0.0
		if min_fairness > 0.0:
			test_fair = 1.0 - fair_test(pipeline, X_test, pd.DataFrame(y_test))
		test_robust = 0.0
		if min_robustness > 0.0:
			test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test,
												  model=pipeline.named_steps['clf'],
												  feature_selector=pipeline.named_steps['selection'],
												  scorer=auc_scorer)

		my_result['test_fair'] = test_fair
		my_result['test_acc'] = test_acc
		my_result['test_robust'] = test_robust
		my_result['final_time'] = time.time() - start_time

		if validation_fair >= min_fairness and validation_acc >= min_accuracy and validation_robust >= min_robustness:
			my_result['Finished'] = True
			my_result['Validation_Satisfied'] = True

			success = False
			if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
				success = True

			cheating_global.successfull_result[hash]['success'] = success

			my_result['success_test'] = success
			pickle.dump(my_result, f_log)
			f_log.close()
			return [validation_acc, validation_fair, validation_robust, validation_simplicity, min_loss]


		if min_loss > loss:
			min_loss = loss
			pickle.dump(my_result, f_log)

		return [validation_acc, validation_fair, validation_robust, validation_simplicity, min_loss]
	
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

			self.min_loss = np.inf

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
				results = objective(x[i], self.min_loss)
				accuracy_batch.append(results[0]*-1)#accuracy
				fairness_batch.append(results[1]*-1)#fairness
				robustness_batch.append(results[2]*-1)#robustness
				simplicity_batch.append(results[3]*-1)  # simplicity
				self.min_loss = results[4]

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
	
	population_size = 30
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
			if 'success' in cheating_global.successfull_result[hash] or (time.time() - self.start_time) > self.time_limit:
				return False
	
			return True

	minimize(problem=problem, algorithm=algorithm, termination=MyTermination(start_time, max_search_time), disp=False)

	number_of_evaluations = cheating_global.successfull_result[hash]['cv_number_evaluations']

	success = False
	if not 'success' in cheating_global.successfull_result[hash]:
		my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time,
					 'Finished': True}
		pickle.dump(my_result, f_log)
		f_log.close()
	else:
		success = copy.deepcopy(cheating_global.successfull_result[hash]['success'])


	del cheating_global.successfull_result[hash]
	return {'success': success}
