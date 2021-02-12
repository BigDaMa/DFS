import copy
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
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.utils.gridsearch import is_utility_defined

def forward_floating_selection(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True),model_hyperparameters=None):
	return forward_floating_selection_lib(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=clf, min_accuracy = min_accuracy, min_fairness=min_fairness, min_robustness=min_robustness, max_number_features=max_number_features, max_search_time=max_search_time, log_file=log_file, floating=True, accuracy_scorer=accuracy_scorer, model_hyperparameters=model_hyperparameters)

def forward_selection(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True),model_hyperparameters=None):
	return forward_floating_selection_lib(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=clf, min_accuracy = min_accuracy, min_fairness=min_fairness, min_robustness=min_robustness, max_number_features=max_number_features, max_search_time=max_search_time, log_file=log_file, floating=False, accuracy_scorer=accuracy_scorer, model_hyperparameters=model_hyperparameters)


def forward_floating_selection_lib(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None, floating=True, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), model_hyperparameters=None):
	min_loss = np.inf
	start_time = time.time()

	fair_validation = None
	fair_test = None
	if type(sensitive_ids) != type(None):
		fair_validation = make_scorer(true_positive_rate_score, greater_is_better=True,
									  sensitive_data=X_validation[:, sensitive_ids[0]])
		fair_test = make_scorer(true_positive_rate_score, greater_is_better=True,
								sensitive_data=X_test[:, sensitive_ids[0]])

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

		model = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', clf)
		])

		return model, hps

	def f_to_min1(hps):
		pipeline, hps = f_clf1(hps)

		if np.sum(pipeline.named_steps['selection'].mask) == 0:
			return {'loss': 4, 'status': STATUS_OK, 'model': pipeline, 'cv_fair': 0.0, 'cv_acc': 0.0, 'cv_robust': 0.0, 'cv_number_features': 1.0}

		grid_result = run_grid_search(pipeline, X_train, y_train, X_validation, y_validation,
									  accuracy_scorer, sensitive_ids,
									  min_fairness, min_accuracy, min_robustness, max_number_features,
									  model_hyperparameters=model_hyperparameters, start_time=start_time)

		grid_result['updated_parameters'] = hps
		return grid_result

	def execute_feature_combo(feature_combo, number_of_evaluations):
		hps = {}
		for f_i in range(X_train.shape[1]):
			if f_i in feature_combo:
				hps['f_' + str(f_i)] = 1
			else:
				hps['f_' + str(f_i)] = 0

		result = f_to_min1(hps)

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

		if cv_fair >= min_fairness and cv_acc >= min_accuracy and cv_robust >= min_robustness and cv_number_features <= max_number_features  and not is_utility_defined(min_fairness, min_accuracy, min_robustness, max_number_features):
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

			return my_result, {'success': success}

		return my_result, {}




	space = {}
	for f_i in range(X_train.shape[1]):
		space['f_' + str(f_i)] = hp.randint('f_' + str(f_i), 2)

	cv_fair = 0
	cv_acc = 0
	cv_robust = 0
	cv_number_features = 1.0

	number_of_evaluations = 0

	max_k = max(int(max_number_features * X_train.shape[1]), 1)

	current_feature_set = []
	remaining_features = list(range(X_train.shape[1]))

	history = {}
	while (len(current_feature_set) < max_k):
		# select best feature
		best_feature_id = -1
		lowest_loss = np.inf
		for new_feature in remaining_features:

			feature_combo = [new_feature]
			feature_combo.extend(current_feature_set)

			# book-keeping to avoid infinite loops
			if frozenset(feature_combo) in history:
				continue
			number_of_evaluations += 1

			my_result, combo_result = execute_feature_combo(feature_combo, number_of_evaluations)
			if min_loss > my_result['loss']:
				min_loss = my_result['loss']
				with open(log_file, 'ab') as f_log:
					my_result_new = copy.deepcopy(my_result)
					my_result_new['selected_features'] = copy.deepcopy(my_result_new['model'].named_steps['selection'])
					my_result_new['model'] = None
					pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)
			if len(combo_result) > 0:
				return combo_result
			combo_loss = my_result['loss']
			#print('FS: ' + str(combo_loss))

			history[frozenset(feature_combo)] = my_result['loss']
			if combo_loss < lowest_loss:
				best_feature_id = new_feature
				lowest_loss = combo_loss

		if best_feature_id == -1:
			break

		current_feature_set.append(best_feature_id)
		remaining_features.remove(best_feature_id)

		if floating:
			# select worst feature
			while True:
				best_feature_id = -1
				lowest_loss_new = np.inf
				for i in range(len(current_feature_set)-1, 0, -1):
					new_feature = current_feature_set[i]
					feature_combo = copy.deepcopy(current_feature_set)
					feature_combo.remove(new_feature)

					#book-keeping to avoid infinite loops
					if frozenset(feature_combo) in history:
						continue
					number_of_evaluations += 1

					my_result, combo_result = execute_feature_combo(feature_combo, number_of_evaluations)
					if min_loss > my_result['loss']:
						min_loss = my_result['loss']
						with open(log_file, 'ab') as f_log:
							my_result_new = copy.deepcopy(my_result)
							my_result_new['selected_features'] = copy.deepcopy(
								my_result_new['model'].named_steps['selection'])
							my_result_new['model'] = None
							pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)
					if len(combo_result) > 0:
						return combo_result
					combo_loss = my_result['loss']
					#print('FS: ' + str(combo_loss))

					history[frozenset(feature_combo)] = combo_loss
					if combo_loss < lowest_loss_new:
						best_feature_id = new_feature
						lowest_loss_new = combo_loss

				if lowest_loss_new > lowest_loss:
					break
				else:
					lowest_loss = lowest_loss_new

					current_feature_set.remove(best_feature_id)
					remaining_features.append(best_feature_id)

	my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time,
				 'Finished': True}
	with open(log_file, 'ab') as f_log:
		my_result_new = copy.deepcopy(my_result)
		pickle.dump(my_result_new, f_log, protocol=pickle.HIGHEST_PROTOCOL)
	return {'success': False}






