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
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.utils.gridsearch import run_grid_search
import copy

def map_hyper2vals(hyper):
	new_vals = {}
	for k, v in hyper.items():
		new_vals[k] = [v]
	return new_vals


def fullfeatures(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features = None, max_search_time=np.inf, log_file=None, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)):
	start_time = time.time()

	fair_validation = None
	fair_test = None
	if type(sensitive_ids) != type(None):
		fair_validation = make_scorer(true_positive_rate_score, greater_is_better=True,
									  sensitive_data=X_validation[:, sensitive_ids[0]])
		fair_test = make_scorer(true_positive_rate_score, greater_is_better=True,
								sensitive_data=X_test[:, sensitive_ids[0]])

	def f_to_min1():
		mask = np.array([True for f_i in range(X_train.shape[1])])

		pipeline = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', clf)
		])

		grid_result = run_grid_search(pipeline, X_train, y_train, X_validation, y_validation,
									  accuracy_scorer, sensitive_ids,
									  min_fairness, min_accuracy, min_robustness, max_number_features,
									  model_hyperparameters=None, start_time=start_time)

		return grid_result

	space = {}
	for f_i in range(X_train.shape[1]):
		space['f_' + str(f_i)] = hp.randint('f_' + str(f_i), 2)

	number_of_evaluations = 0

	result = f_to_min1()

	number_of_evaluations += 1

	cv_fair = result['cv_fair']
	cv_acc = result['cv_acc']
	cv_robust = result['cv_robust']
	cv_number_features = result['cv_number_features']

	my_result = result

	my_result['number_evaluations'] = number_of_evaluations

	model = result['model']
	model.fit(X_train_val, pd.DataFrame(y_train_val))

	test_acc = accuracy_scorer(model, X_test, pd.DataFrame(y_test))

	test_fair = 0.0
	if type(sensitive_ids) != type(None):
		test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
	test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test,
										  model=model.named_steps['clf'],
										  feature_selector=model.named_steps['selection'],
										  scorer=accuracy_scorer)

	my_result['test_fair'] = test_fair
	my_result['test_acc'] = test_acc
	my_result['test_robust'] = test_robust
	my_result['final_time'] = time.time() - start_time

	if cv_fair >= min_fairness and cv_acc >= min_accuracy and cv_robust >= min_robustness and cv_number_features <= max_number_features:
		my_result['Finished'] = True
		my_result['Validation_Satisfied'] = True

		success = False
		if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
			success = True

		my_result['success_test'] = success
		with open(log_file, 'ab') as f_log:
			my_result_new = copy.deepcopy(my_result)
			my_result_new['selected_features'] = copy.deepcopy(my_result_new['model'].named_steps['selection'])
			my_result_new['model'] = None
			pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)
		return {'success': success}

	with open(log_file, 'ab') as f_log:
		my_result_new = copy.deepcopy(my_result)
		my_result_new['selected_features'] = copy.deepcopy(my_result_new['model'].named_steps['selection'])
		my_result_new['model'] = None
		pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)


	my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time,
				 'Finished': True}
	with open(log_file, 'ab') as f_log:
		my_result_new = copy.deepcopy(my_result)
		pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)
	return {'success': False}






