from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
from hyperopt.pyll.base import scope
import numpy as np
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.feature_selection.WeightedRankingSelection import WeightedRankingSelection



def weighted_ranking(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features: float = 1.0, max_search_time=np.inf, log_file=None):
	min_loss = np.inf
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

	#calculate rankings
	rankings = []
	for ranking_function_i in range(len(ranking_functions)):
		rankings.append(ranking_functions[ranking_function_i](X_train, y_train))

	def f_clf1(hps):
		weights = []
		for i in range(len(rankings)):
			weights.append(hps['weight' + str(i)])

		pipeline = Pipeline([
			('selection', WeightedRankingSelection(scores=rankings, weights=weights, k=hps['k'], names=np.array(names))),
			('clf', clf)
		])

		return pipeline

	stored_results = {}
	fail_counter = [0]

	def f_to_min1(hps):
		if str(hps) in stored_results:
			fail_counter[0] += 1
			return stored_results[str(hps)]
		fail_counter[0] = 0


		pipeline = f_clf1(hps)
		pipeline.fit(X_train, pd.DataFrame(y_train))


		validation_number_features = float(np.sum(pipeline.named_steps['selection']._get_support_mask())) / float(X_train.shape[1])

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

		current_time = time.time() - start_time

		stored_results[str(hps)] = {'loss': loss,
									'status': STATUS_OK,
									'model': pipeline,
									'cv_fair': validation_fair,
									'cv_acc': validation_acc,
									'cv_robust': validation_robust,
									'cv_number_features': validation_number_features,
									'time': current_time,
									'updated_parameters': hps}

		return stored_results[str(hps)]



	max_k = max(int(max_number_features * X_train.shape[1]), 1)
	space = {'k': scope.int(hp.quniform('k', 1, max_k, 1))}

	if len(rankings) > 1:
		for i in range(len(rankings)):
			space['weight' + str(i)] = hp.choice('weight' + str(i) + 'choice',
								  [
									  (0.0),
									  hp.lognormal('weight' + str(i) + 'specified', 0, 1)
								  ])
	else:
		space['weight' + str(0)] = 1.0

	number_of_evaluations = 0

	trials = Trials()
	i = 1
	success = False
	while True:
		if time.time() - start_time > max_search_time:
			break
		if len(stored_results) >= max_k:
			break
		if fail_counter[0] >= 10:
			break

		fmin(f_to_min1, space=space, algo=tpe.suggest, max_evals=i, trials=trials)

		number_of_evaluations += 1

		cv_fair = trials.trials[-1]['result']['cv_fair']
		validation_acc = trials.trials[-1]['result']['cv_acc']
		validation_robust = trials.trials[-1]['result']['cv_robust']

		my_result = trials.trials[-1]['result']
		my_result['number_evaluations'] = number_of_evaluations

		#check test results
		model = trials.trials[-1]['result']['model']
		model.fit(X_train_val, pd.DataFrame(y_train_val))

		test_acc = 0.0
		if min_accuracy > 0.0:
			test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
		test_fair = 0.0
		if min_fairness > 0.0:
			test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
		test_robust = 0.0
		if min_robustness > 0.0:
			test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test,
												  model=model.named_steps['clf'],
												  feature_selector=model.named_steps['selection'],
												  scorer=auc_scorer)

		my_result['test_fair'] = test_fair
		my_result['test_acc'] = test_acc
		my_result['test_robust'] = test_robust
		my_result['final_time'] = time.time() - start_time

		if cv_fair >= min_fairness and validation_acc >= min_accuracy and validation_robust >= min_robustness:
			my_result['Finished'] = True
			my_result['Validation_Satisfied'] = True

			if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
				success = True

			my_result['success_test'] = success
			pickle.dump(my_result, f_log)
			f_log.close()
			return {'success': success}


		if min_loss > trials.trials[-1]['result']['loss']:
			min_loss = trials.trials[-1]['result']['loss']
			pickle.dump(my_result, f_log)

		i += 1

	my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time, 'Finished': True}
	pickle.dump(my_result, f_log)
	f_log.close()
	return {'success': False}






