from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.utils.gridsearch import run_grid_search

from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
import hyperopt.anneal
import copy
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.utils.gridsearch import is_utility_defined

def map_hyper2vals(hyper):
	new_vals = {}
	for k, v in hyper.items():
		new_vals[k] = [v]
	return new_vals

def TPE(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features = None, max_search_time=np.inf, log_file = None, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), avoid_robustness=False, model_hyperparameters=None):
	return hyperparameter_optimization(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions=[],
									   clf=clf, min_accuracy=min_accuracy, min_fairness=min_fairness,
									   min_robustness=min_robustness, max_number_features=max_number_features,
									   max_search_time=max_search_time, log_file=log_file,
									   algo=tpe.suggest, accuracy_scorer=accuracy_scorer, avoid_robustness=avoid_robustness, model_hyperparameters=model_hyperparameters)


def simulated_annealing(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions=[], clf=None, min_accuracy=0.0,
			min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), avoid_robustness=False, model_hyperparameters=None):

	return hyperparameter_optimization(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions=[], clf=clf, min_accuracy=min_accuracy, min_fairness=min_fairness, min_robustness=min_robustness, max_number_features=max_number_features, max_search_time=max_search_time, log_file=log_file, algo=hyperopt.anneal.suggest, accuracy_scorer=accuracy_scorer, avoid_robustness=avoid_robustness, model_hyperparameters=model_hyperparameters)



def hyperparameter_optimization(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features = None, max_search_time=np.inf, log_file=None, algo=tpe.suggest, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), avoid_robustness=False, model_hyperparameters=None):
	min_loss = np.inf
	start_time = time.time()

	fair_validation = None
	fair_test = None
	if type(sensitive_ids) != type(None):
		fair_validation = make_scorer(true_positive_rate_score, greater_is_better=True,
									  sensitive_data=X_validation[:, sensitive_ids[0]])
		fair_test = make_scorer(true_positive_rate_score, greater_is_better=True,
								sensitive_data=X_test[:, sensitive_ids[0]])

	stored_results = {} #avoid duplicate hyperparameters

	def f_clf1(hps):
		mask = np.zeros(len(hps), dtype=bool)
		for k, v in hps.items():
			mask[int(k.split('_')[1])] = v


		#repair number of features if neccessary
		max_k = max(int(max_number_features * X_train.shape[1]), 1)
		if np.sum(mask) > max_k:
			id_features_used = np.nonzero(mask)[0]  # indices where features are used
			np.random.shuffle(id_features_used)  # shuffle ids
			ids_tb_deactived = id_features_used[max_k:]  # deactivate features
			for item_to_remove in ids_tb_deactived:
				mask[item_to_remove] = False

		for mask_i in range(len(mask)):
			hps['f_' + str(mask_i)] = mask[mask_i]

		pipeline = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', clf)
		])

		return pipeline, hps

	def f_to_min1(hps):
		if str(hps) in stored_results:
			return stored_results[str(hps)]

		pipeline, hps = f_clf1(hps)

		if np.sum(pipeline.named_steps['selection'].mask) == 0:
			stored_results[str(hps)] = {'loss': 4, 'status': STATUS_OK, 'model': pipeline, 'cv_fair': 0.0, 'cv_acc': 0.0, 'cv_robust': 0.0, 'cv_number_features': 1.0}
			return stored_results[str(hps)]

		stored_results[str(hps)] = run_grid_search(pipeline, X_train, y_train, X_validation, y_validation, accuracy_scorer, sensitive_ids,
						min_fairness, min_accuracy, min_robustness, max_number_features, model_hyperparameters=model_hyperparameters, start_time=start_time, avoid_robustness=avoid_robustness)

		stored_results[str(hps)]['updated_parameters'] = hps

		return stored_results[str(hps)]

	space = {}
	for f_i in range(X_train.shape[1]):
		space['f_' + str(f_i)] = hp.randint('f_' + str(f_i), 2)

	number_of_evaluations = 0

	trials = Trials()
	i = 1
	success = False
	fail_counter = [0]

	while True:
		if time.time() - start_time > max_search_time:
			break
		if fail_counter[0] >= 10:
			fail_counter[0] += 1
			break
		fail_counter[0] = 0

		fmin(f_to_min1, space=space, algo=algo, max_evals=i, trials=trials)

		#update repair in database
		try:
			current_trial = trials.trials[-1]
			if type(current_trial['result']['updated_parameters']) != type(None):
				trials._dynamic_trials[-1]['misc']['vals'] = map_hyper2vals(current_trial['result']['updated_parameters'])
		except:
			print("found an error in repair")


		number_of_evaluations += 1

		cv_fair = trials.trials[-1]['result']['cv_fair']
		validation_acc = trials.trials[-1]['result']['cv_acc']
		validation_robust = trials.trials[-1]['result']['cv_robust']
		cv_number_features = trials.trials[-1]['result']['cv_number_features']

		my_result = trials.trials[-1]['result']

		my_result['number_evaluations'] = number_of_evaluations


		model = trials.trials[-1]['result']['model']
		model.fit(X_train_val, pd.DataFrame(y_train_val))

		test_acc = accuracy_scorer(model, X_test, pd.DataFrame(y_test))
		test_fair = 0.0
		if type(sensitive_ids) != type(None):
			test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))

		test_robust = 0.0
		if not avoid_robustness:
			test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test,
												  model=model.named_steps['clf'],
												  feature_selector=model.named_steps['selection'],
												  scorer=accuracy_scorer)

		my_result['test_fair'] = test_fair
		my_result['test_acc'] = test_acc
		my_result['test_robust'] = test_robust
		my_result['final_time'] = time.time() - start_time

		if cv_fair >= min_fairness and validation_acc >= min_accuracy and validation_robust >= min_robustness and cv_number_features <= max_number_features and not is_utility_defined(min_fairness, min_accuracy, min_robustness, max_number_features):
			my_result['Finished'] = True
			my_result['Validation_Satisfied'] = True

			if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
				success = True

			my_result['success_test'] = success
			with open(log_file, 'ab') as f_log:
				my_result_new = copy.deepcopy(my_result)
				my_result_new['selected_features'] = copy.deepcopy(my_result_new['model'].named_steps['selection'])
				my_result_new['model'] = None
				pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)
			return {'success': success}

		if min_loss > trials.trials[-1]['result']['loss']:
			min_loss = trials.trials[-1]['result']['loss']
			with open(log_file, 'ab') as f_log:
				my_result_new = copy.deepcopy(my_result)
				my_result_new['selected_features'] = copy.deepcopy(my_result_new['model'].named_steps['selection'])
				my_result_new['model'] = None
				pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)


		i += 1

	my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time,
				 'Finished': True}
	with open(log_file, 'ab') as f_log:
		my_result_new = copy.deepcopy(my_result)
		pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)
	return {'success': False}





