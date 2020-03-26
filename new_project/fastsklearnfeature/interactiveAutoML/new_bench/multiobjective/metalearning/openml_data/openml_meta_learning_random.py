import copy
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_mcfs
from sklearn.model_selection import train_test_split
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score


from fastsklearnfeature.configuration.Config import Config



from functools import partial
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from skrebate import ReliefF
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.multiprocessing_global as mp_global
import diffprivlib.models as models
from sklearn.model_selection import GridSearchCV

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


#static constraints: fairness, number of features (absolute and relative), robustness, privacy, accuracy

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_fair_data1
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

current_run_time_id = time.time()

time_limit = 60 * 60 * 3
n_jobs = 20
number_of_runs = 1

X_train_meta_classifier = []
y_train_meta_classifier = []

ranking_scores_info = []


acc_value_list = []
fair_value_list = []
robust_value_list = []
success_value_list = []
runtime_value_list = []
evaluation_value_list = []
k_value_list = []

dataset_did_list = []
constraint_set_list = []
dataset_sensitive_attribute_list = []

cv_splitter = StratifiedKFold(5, random_state=42)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

while True:
	X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id = get_fair_data1()

	#run on tiny sample
	if X_train.shape[0] > 100:
		X_train_tiny, _, y_train_tiny, _ = train_test_split(X_train, y_train, train_size=100, random_state=42, stratify=y_train)
	else:
		X_train_tiny = X_train
		y_train_tiny = y_train

	fair_train_tiny = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train_tiny[:, sensitive_ids[0]])

	mp_global.X_train = X_train
	mp_global.X_test = X_test
	mp_global.y_train = y_train
	mp_global.y_test = y_test
	mp_global.names = names
	mp_global.sensitive_ids = sensitive_ids
	mp_global.cv_splitter = cv_splitter


	def objective(hps):
		print(hps)

		try:
			cv_k = 1.0
			cv_privacy = hps['privacy']
			model = LogisticRegression()
			if type(cv_privacy) == type(None):
				cv_privacy = X_train_tiny.shape[0]
			else:
				model = models.LogisticRegression(epsilon=cv_privacy)

			robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train_tiny, y=y_train_tiny, model=model,
										feature_selector=None, scorer=auc_scorer)


			small_start_time = time.time()

			cv = GridSearchCV(model, param_grid={'C': [1.0]}, scoring={'AUC': auc_scorer, 'Fairness': fair_train_tiny, 'Robustness': robust_scorer}, refit=False, cv=cv_splitter)
			cv.fit(X_train_tiny, pd.DataFrame(y_train_tiny))
			cv_acc = cv.cv_results_['mean_test_AUC'][0]
			cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]
			cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

			cv_time = time.time() - small_start_time

			##apply rankings
			scores_stored = []
			my_rankings_on_tiny = [variance, chi2_score_wo, my_fisher_score]
			for rr_i in range(len(my_rankings_on_tiny)):
				start_scoring_tiny = time.time()
				scores_tiny = my_rankings_on_tiny[rr_i](X_train_tiny, y_train_tiny)
				scores_stored.append({'name': my_rankings_on_tiny[rr_i].__name__, 'scores': scores_tiny, 'time': time.time() - start_scoring_tiny})




			#construct feature vector
			feature_list = []
			#user-specified constraints
			feature_list.append(hps['accuracy'])
			feature_list.append(hps['fairness'])
			feature_list.append(hps['k'])
			feature_list.append(hps['k'] * X_train.shape[1])
			feature_list.append(hps['robustness'])
			feature_list.append(cv_privacy)
			feature_list.append(hps['search_time'])
			#differences to sample performance
			feature_list.append(cv_acc - hps['accuracy'])
			feature_list.append(cv_fair - hps['fairness'])
			feature_list.append(cv_k - hps['k'])
			feature_list.append((cv_k - hps['k']) * X_train.shape[1])
			feature_list.append(cv_robust - hps['robustness'])
			feature_list.append(cv_time)
			#privacy constraint is always satisfied => difference always zero => constant => unnecessary

			#metadata features
			feature_list.append(X_train.shape[0])#number rows
			feature_list.append(X_train.shape[1])#number columns

			features = np.array(feature_list)

			#predict the best model and calculate uncertainty

			loss = 0
			return {'loss': loss, 'status': STATUS_OK, 'features': features, 'search_time': hps['search_time'], 'ranking_scores': scores_stored, 'constraints': hps }
		except:
			return {'loss': np.inf, 'status': STATUS_OK}



	space = {
			 'k': hp.choice('k_choice',
							[
								(1.0),
								(hp.uniform('k_specified', 0, 1))
							]),
			 'accuracy': hp.uniform('accuracy_specified', 0.5, 1),
			 'fairness': hp.choice('fairness_choice',
							[
								(0.0),
								(hp.uniform('fairness_specified', 0, 1))
							]),
			 'privacy': hp.choice('privacy_choice',
							[
								(None),
								(hp.lognormal('privacy_specified', 0, 1))
							]),
			 'robustness': hp.choice('robustness_choice',
							[
								(0.0),
								(hp.uniform('robustness_specified', 0, 1))
							]),
		     'search_time': hp.uniform('search_time_specified', 10, time_limit
									   ), # in seconds
			}

	trials = Trials()
	runs_per_dataset = 0
	i = 1
	while True:
		fmin(objective, space=space, algo=tpe.suggest, max_evals=i, trials=trials, show_progressbar=False)
		i += 1

		if trials.trials[-1]['result']['loss'] == np.inf:
			break

		#break, once convergence tolerance is reached and generate new dataset
		best_trial = trials.trials[-1]
		most_uncertain_f = best_trial['misc']['vals']
		#print(most_uncertain_f)

		min_accuracy = most_uncertain_f['accuracy_specified'][0]
		min_fairness = 0.0
		if most_uncertain_f['fairness_choice'][0]:
			min_fairness = most_uncertain_f['fairness_specified'][0]
		min_robustness = 0.0
		if most_uncertain_f['robustness_choice'][0]:
			min_robustness = most_uncertain_f['robustness_specified'][0]
		max_number_features = 1.0
		if most_uncertain_f['k_choice'][0]:
			max_number_features = most_uncertain_f['k_specified'][0]

		max_search_time = most_uncertain_f['search_time_specified'][0]

		# Execute each search strategy with a given time limit (in parallel)
		# maybe run multiple times to smooth stochasticity

		model = LogisticRegression()
		if most_uncertain_f['privacy_choice'][0]:
			model = models.LogisticRegression(epsilon=most_uncertain_f['privacy_specified'][0])
		mp_global.clf = model

		#define rankings
		rankings = [variance,
					chi2_score_wo,
					fcbf,
					my_fisher_score,
					mutual_info_classif,
					my_mcfs]
		#rankings.append(partial(model_score, estimator=ExtraTreesClassifier(n_estimators=1000))) #accuracy ranking
		#rankings.append(partial(robustness_score, model=model, scorer=auc_scorer)) #robustness ranking
		#rankings.append(partial(fairness_score, estimator=ExtraTreesClassifier(n_estimators=1000), sensitive_ids=sensitive_ids)) #fairness ranking
		rankings.append(partial(model_score, estimator=ReliefF(n_neighbors=10)))  # relieff

		mp_global.min_accuracy = min_accuracy
		mp_global.min_fairness = min_fairness
		mp_global.min_robustness = min_robustness
		mp_global.max_number_features = max_number_features
		mp_global.max_search_time = max_search_time

		mp_global.configurations = []
		#add single rankings
		strategy_id = 1
		for r in range(len(rankings)):
			for run in range(number_of_runs):
				configuration = {}
				configuration['ranking_functions'] = copy.deepcopy([rankings[r]])
				configuration['run_id'] = copy.deepcopy(run)
				configuration['main_strategy'] = copy.deepcopy(weighted_ranking)
				configuration['strategy_id'] = copy.deepcopy(strategy_id)
				mp_global.configurations.append(configuration)
			strategy_id +=1

		main_strategies = [TPE,
						   simulated_annealing,
						   evolution,
						   exhaustive,
						   forward_selection,
						   backward_selection,
						   forward_floating_selection,
						   backward_floating_selection,
						   recursive_feature_elimination]

		#run main strategies
		for strategy in main_strategies:
			for run in range(number_of_runs):
					configuration = {}
					configuration['ranking_functions'] = []
					configuration['run_id'] = copy.deepcopy(run)
					configuration['main_strategy'] = copy.deepcopy(strategy)
					configuration['strategy_id'] = copy.deepcopy(strategy_id)
					mp_global.configurations.append(configuration)
			strategy_id += 1

		def my_function(config_id):
			conf = mp_global.configurations[config_id]
			result = conf['main_strategy'](mp_global.X_train, mp_global.X_test, mp_global.y_train, mp_global.y_test, mp_global.names, mp_global.sensitive_ids,
						 ranking_functions=conf['ranking_functions'],
						 clf=mp_global.clf,
						 min_accuracy=mp_global.min_accuracy,
						 min_fairness=mp_global.min_fairness,
						 min_robustness=mp_global.min_robustness,
						 max_number_features=mp_global.max_number_features,
						 max_search_time=mp_global.max_search_time,
						 cv_splitter=mp_global.cv_splitter)
			result['strategy_id'] = conf['strategy_id']
			return result


		results = []
		check_strategies = np.zeros(strategy_id)
		with ProcessPool(max_workers=16) as pool:
			future = pool.map(my_function, range(len(mp_global.configurations)), timeout=max_search_time)

			iterator = future.result()
			while True:
				try:
					result = next(iterator)
					check_strategies[result['strategy_id']] += 1
					results.append(result)
				except StopIteration:
					break
				except TimeoutError as error:
					print("function took longer than %d seconds" % error.args[1])
				except ProcessExpired as error:
					print("%s. Exit code: %d" % (error, error.exitcode))
				except Exception as error:
					print("function raised %s" % error)
					print(error.traceback)  # Python's traceback of remote process
		results.append({'strategy_id': 0, 'time': np.inf, 'success': True})  # none of the strategies reached the constraint

		#average runtime for each method
		runtimes = np.zeros(strategy_id)
		success = np.zeros(strategy_id, dtype=bool)

		accuracy_values = {}
		fairness_values = {}
		robustness_values = {}
		k_values = {}
		success_values = {}
		runtime_values = {}
		evaluation_values = {}

		for r in range(len(results)):
			runtimes[results[r]['strategy_id']] += results[r]['time']
			if results[r]['strategy_id'] > 0:
				if not results[r]['strategy_id'] in accuracy_values:
					accuracy_values[results[r]['strategy_id']] = []
					fairness_values[results[r]['strategy_id']] = []
					robustness_values[results[r]['strategy_id']] = []
					k_values[results[r]['strategy_id']] = []
					success_values[results[r]['strategy_id']] = []
					runtime_values[results[r]['strategy_id']] = []
					evaluation_values[results[r]['strategy_id']] = []

				#add stuff
				if 'cv_acc' in results[r]:
					accuracy_values[results[r]['strategy_id']].append(results[r]['cv_acc'])
				else:
					accuracy_values[results[r]['strategy_id']].append(-1)

				if 'cv_fair' in results[r]:
					fairness_values[results[r]['strategy_id']].append(results[r]['cv_fair'])
				else:
					fairness_values[results[r]['strategy_id']].append(-1)

				if 'cv_robust' in results[r]:
					robustness_values[results[r]['strategy_id']].append(results[r]['cv_robust'])
				else:
					robustness_values[results[r]['strategy_id']].append(-1)

				if 'cv_number_features' in results[r]:
					k_values[results[r]['strategy_id']].append(results[r]['cv_number_features'])
				else:
					k_values[results[r]['strategy_id']].append(-1)

				if 'success' in results[r]:
					success_values[results[r]['strategy_id']].append(results[r]['success'])
				else:
					success_values[results[r]['strategy_id']].append(-1)

				if 'time' in results[r]:
					runtime_values[results[r]['strategy_id']].append(results[r]['time'])
				else:
					runtime_values[results[r]['strategy_id']].append(-1)

				if 'cv_number_evaluations' in results[r]:
					evaluation_values[results[r]['strategy_id']].append(results[r]['cv_number_evaluations'])
				else:
					evaluation_values[results[r]['strategy_id']].append(-1)
			if results[r]['success']:
				success[results[r]['strategy_id']] = True


		for strategy_i in range(1, strategy_id):
			number_successes = 0
			if strategy_i in success_values:
				number_successes += np.sum(success_values[strategy_i])
			runtimes[strategy_i] += (number_of_runs - number_successes) * mp_global.max_search_time

		#get lowest runtime
		ids = np.argsort(runtimes)
		best_strategy = -1
		for id_i in range(len(ids)):
			if success[ids[id_i]]:
				best_strategy = ids[id_i]
				break
		print('best strategy: ' + str(best_strategy))

		# append ml data
		X_train_meta_classifier.append(best_trial['result']['features'])
		y_train_meta_classifier.append(best_strategy)

		ranking_scores_info.append(best_trial['result']['ranking_scores'])

		#pickle everything and store it
		one_big_object = {}
		one_big_object['features'] = X_train_meta_classifier
		one_big_object['best_strategy'] = y_train_meta_classifier
		one_big_object['ranking_scores_info'] = ranking_scores_info

		runtime_value_list.append(runtime_values)
		acc_value_list.append(accuracy_values)
		fair_value_list.append(fairness_values)
		robust_value_list.append(robustness_values)
		success_value_list.append(success_values)
		evaluation_value_list.append(evaluation_values)
		k_value_list.append(k_values)

		dataset_did_list.append(data_did)
		dataset_sensitive_attribute_list.append(sensitive_attribute_id)
		constraint_set_list.append(trials.trials[-1]['result']['constraints'])

		one_big_object['times_value'] = runtime_value_list
		one_big_object['k_value'] = k_value_list
		one_big_object['acc_value'] = acc_value_list
		one_big_object['fair_value'] = fair_value_list
		one_big_object['robust_value'] = robust_value_list
		one_big_object['success_value'] = success_value_list
		one_big_object['evaluation_value'] = evaluation_value_list
		one_big_object['dataset_id'] = dataset_did_list
		one_big_object['constraint_set_list'] = constraint_set_list
		one_big_object['sensitive_attribute_id'] = dataset_sensitive_attribute_list

		pickle.dump(one_big_object, open('/tmp/metalearning_data' + str(current_run_time_id) + '.pickle', 'wb'))

		trials = Trials()
		i = 1
		runs_per_dataset += 1
		break


