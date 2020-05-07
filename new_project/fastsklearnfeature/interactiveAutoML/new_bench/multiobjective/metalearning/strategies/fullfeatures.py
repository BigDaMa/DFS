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


def map_hyper2vals(hyper):
	new_vals = {}
	for k, v in hyper.items():
		new_vals[k] = [v]
	return new_vals


def fullfeatures(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features = None, max_search_time=np.inf, log_file=None):
	start_time = time.time()

	auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

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

		pipeline.fit(X_train, y_train)

		validation_number_features = 1.0
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
		print('full: ' + str(loss))

		current_time = time.time() - start_time

		return {'loss': loss,
				'model': pipeline,
				'cv_fair': validation_fair,
				'cv_acc': validation_acc,
				'cv_robust': validation_robust,
				'cv_number_features': validation_number_features,
				'time': current_time}

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
		f_log = open(log_file, 'ab')
		pickle.dump(my_result, f_log, protocol=pickle.HIGHEST_PROTOCOL)
		f_log.close()
		return {'success': success}

	f_log = open(log_file, 'ab')
	pickle.dump(my_result, f_log, protocol=pickle.HIGHEST_PROTOCOL)
	f_log.close()
	my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time,
				 'Finished': True}
	f_log = open(log_file, 'ab')
	pickle.dump(my_result, f_log, protocol=pickle.HIGHEST_PROTOCOL)
	f_log.close()
	return {'success': False}






