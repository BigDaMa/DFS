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
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.utils.gridsearch import run_grid_search


def weighted_ranking(X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features: float = 1.0, max_search_time=np.inf, log_file=None, accuracy_scorer=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)):
	min_loss = np.inf

	start_time = time.time()

	fair_validation = None
	fair_test = None
	if type(sensitive_ids) != type(None):
		fair_validation = make_scorer(true_positive_rate_score, greater_is_better=True,
									  sensitive_data=X_validation[:, sensitive_ids[0]])
		fair_test = make_scorer(true_positive_rate_score, greater_is_better=True,
								sensitive_data=X_test[:, sensitive_ids[0]])

	#calculate rankings
	try:
		rankings = []
		for ranking_function_i in range(len(ranking_functions)):
			rankings.append(ranking_functions[ranking_function_i](X_train, y_train))
	except Exception as e:
		my_result = {'error': e}
		#with open(log_file, 'ab') as f_log:
		#	pickle.dump(my_result, f_log, protocol=pickle.HIGHEST_PROTOCOL)
		return {'success': False}

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
		stored_results[str(hps)] = run_grid_search(pipeline, X_train, y_train, X_validation, y_validation,
												   accuracy_scorer, sensitive_ids,
												   min_fairness, min_accuracy, min_robustness, max_number_features,
												   model_hyperparameters=None, start_time=start_time)

		stored_results[str(hps)]['updated_parameters'] = hps

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

		if cv_fair >= min_fairness and validation_acc >= min_accuracy and validation_robust >= min_robustness:
			my_result['Finished'] = True
			my_result['Validation_Satisfied'] = True

			if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
				success = True

			my_result['success_test'] = success
			with open(log_file, 'ab') as f_log:
				pickle.dump(my_result, f_log, protocol=pickle.HIGHEST_PROTOCOL)
			return {'success': success}


		if min_loss > trials.trials[-1]['result']['loss']:
			min_loss = trials.trials[-1]['result']['loss']
			with open(log_file, 'ab') as f_log:
				pickle.dump(my_result, f_log, protocol=pickle.HIGHEST_PROTOCOL)

		i += 1

	my_result = {'number_evaluations': number_of_evaluations, 'success_test': False, 'final_time': time.time() - start_time, 'Finished': True}
	with open(log_file, 'ab') as f_log:
		pickle.dump(my_result, f_log, protocol=pickle.HIGHEST_PROTOCOL)
	return {'success': False}






