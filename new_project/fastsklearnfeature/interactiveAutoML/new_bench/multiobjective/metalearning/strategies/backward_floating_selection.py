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


def backward_floating_selection(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None):
	return backward_floating_selection_lib(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=clf, min_accuracy = min_accuracy, min_fairness=min_fairness, min_robustness=min_robustness, max_number_features=max_number_features, max_search_time=max_search_time, log_file=log_file, floating=True)
def backward_selection(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None):
	return backward_floating_selection_lib(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=clf, min_accuracy = min_accuracy, min_fairness=min_fairness, min_robustness=min_robustness, max_number_features=max_number_features, max_search_time=max_search_time, log_file=log_file, floating=False)


def backward_floating_selection_lib(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness=0.0, min_robustness=0.0, max_number_features=None, max_search_time=np.inf, log_file=None, floating=True):
	f_log = open(log_file, 'wb+')
	min_loss = np.inf
	start_time = time.time()

	auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

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

		pipeline.fit(X_train, pd.DataFrame(y_train))

		validation_number_features = float(np.sum(pipeline.named_steps['selection']._get_support_mask())) / float(
			X_train.shape[1])
		validation_acc = auc_scorer(pipeline, X_validation, pd.DataFrame(y_validation))

		validation_fair = 0.0
		if type(sensitive_ids) != type(None):
			validation_fair = 1.0 - fair_validation(pipeline, X_validation, pd.DataFrame(y_validation))

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

		current_time = time.time() - start_time

		return {'loss': loss,
				'status': STATUS_OK,
				'model': pipeline,
				'cv_fair': validation_fair,
				'cv_acc': validation_acc,
				'cv_robust': validation_robust,
				'cv_number_features': validation_number_features,
				'time': current_time,
				'updated_parameters': hps}

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

		test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
		test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
		test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test,
											  model=model.named_steps['clf'],
											  feature_selector=model.named_steps['selection'],
											  scorer=auc_scorer)

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
			pickle.dump(my_result, f_log)

			return my_result, {'success': success}

		return my_result, {}




	space = {}
	for f_i in range(X_train.shape[1]):
		space['f_' + str(f_i)] = hp.randint('f_' + str(f_i), 2)

	number_of_evaluations = 0

	current_feature_set = list(range(X_train.shape[1]))
	removed_features = []

	history = {}
	while len(current_feature_set) > 1:
		# select best feature
		best_feature_id = -1
		lowest_loss = np.inf
		for new_feature in current_feature_set:

			feature_combo = copy.deepcopy(current_feature_set)
			feature_combo.remove(new_feature)

			# book-keeping to avoid infinite loops
			if frozenset(feature_combo) in history:
				continue
			number_of_evaluations += 1
			my_result, combo_result = execute_feature_combo(feature_combo, number_of_evaluations)
			print('BS: ' + str(my_result['loss']))
			if min_loss > my_result['loss']:
				min_loss = my_result['loss']
				pickle.dump(my_result, f_log)
			if len(combo_result) > 0:
				f_log.close()
				return combo_result
			combo_loss = my_result['loss']

			history[frozenset(feature_combo)] = combo_loss
			if combo_loss < lowest_loss:
				best_feature_id = new_feature
				lowest_loss = combo_loss

		if best_feature_id == -1:
			break
		current_feature_set.remove(best_feature_id)
		removed_features.append(best_feature_id)

		if floating:
			# select worst feature
			while True:
				best_feature_id = -1
				lowest_loss_new = np.inf
				for i in range(len(removed_features)-1,-1,-1):
					new_feature = removed_features[i]
					feature_combo = copy.deepcopy(current_feature_set)
					feature_combo.append(new_feature)

					#book-keeping to avoid infinite loops
					if frozenset(feature_combo) in history:
						continue
					number_of_evaluations += 1
					my_result, combo_result = execute_feature_combo(feature_combo, number_of_evaluations)
					print('BS: ' + str(my_result['loss']))
					if min_loss > my_result['loss']:
						min_loss = my_result['loss']
						pickle.dump(my_result, f_log)
					if len(combo_result) > 0:
						f_log.close()
						return combo_result
					combo_loss = my_result['loss']

					history[frozenset(feature_combo)] = combo_loss
					if combo_loss < lowest_loss_new:
						best_feature_id = new_feature
						lowest_loss_new = combo_loss

				if lowest_loss_new > lowest_loss:
					break
				else:
					lowest_loss = lowest_loss_new

					current_feature_set.append(best_feature_id)
					removed_features.remove(best_feature_id)

	my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time,
				 'Finished': True}
	pickle.dump(my_result, f_log)
	f_log.close()
	return {'success': False}





