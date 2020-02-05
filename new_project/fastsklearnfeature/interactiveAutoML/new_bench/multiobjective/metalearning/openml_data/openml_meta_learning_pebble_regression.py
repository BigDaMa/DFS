import copy
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fairness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import robustness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from fastsklearnfeature.configuration.Config import Config
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score

from sklearn.ensemble import RandomForestRegressor
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.multiprocessing_global as mp_global
import diffprivlib.models as models
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import hyperparameter_optimization
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.evolution import evolution
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_data_openml
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

#load list of viable datasets

def run_strategy(strategy_method, ranking_id, strategy_id):
	data_infos = pickle.load(open(Config.get('data_path') + '/openml_data/fitting_datasets.pickle', 'rb'))

	time_limit = 60 * 20

	meta_classifier = RandomForestRegressor(n_estimators=1000)
	X_train_meta_classifier = []
	y_train_meta_classifier = []

	cv_splitter = StratifiedKFold(5, random_state=42)
	auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


	while True:
		X_train, X_test, y_train, y_test, names, sensitive_ids = get_data_openml(data_infos)

		#run on tiny sample
		X_train_tiny, _, y_train_tiny, _ = train_test_split(X_train, y_train, train_size=100, random_state=42, stratify=y_train)

		fair_train_tiny = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train_tiny[:, sensitive_ids[0]])

		def objective(hps):
			print(hps)

			cv_k = 1.0
			cv_privacy = hps['privacy']
			model = LogisticRegression()
			if type(cv_privacy) == type(None):
				cv_privacy = X_train_tiny.shape[0]
			else:
				model = models.LogisticRegression(epsilon=cv_privacy)

			robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train_tiny, y=y_train_tiny, model=model,
										feature_selector=None, scorer=auc_scorer)


			cv = GridSearchCV(model, param_grid={'C': [1.0]}, scoring={'AUC': auc_scorer, 'Fairness': fair_train_tiny, 'Robustness': robust_scorer}, refit=False, cv=cv_splitter)
			cv.fit(X_train_tiny, pd.DataFrame(y_train_tiny))
			cv_acc = cv.cv_results_['mean_test_AUC'][0]
			cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]
			cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

			#construct feature vector
			feature_list = []
			#user-specified constraints
			feature_list.append(hps['accuracy'])
			feature_list.append(hps['fairness'])
			feature_list.append(hps['k'])
			feature_list.append(hps['k'] * X_train.shape[1])
			feature_list.append(hps['robustness'])
			feature_list.append(cv_privacy)
			#differences to sample performance
			feature_list.append(cv_acc - hps['accuracy'])
			feature_list.append(cv_fair - hps['fairness'])
			feature_list.append(cv_k - hps['k'])
			feature_list.append((cv_k - hps['k']) * X_train.shape[1])
			feature_list.append(cv_robust - hps['robustness'])
			#privacy constraint is always satisfied => difference always zero => constant => unnecessary

			#metadata features
			feature_list.append(X_train.shape[0])#number rows
			feature_list.append(X_train.shape[1])#number columns

			features = np.array(feature_list)

			#predict the best model and calculate uncertainty

			loss = 0
			if hasattr(meta_classifier, 'estimators_'):
				predictions = []
				for tree in range(len(meta_classifier.estimators_)):
					predictions.append(meta_classifier.estimators_[tree].predict([features])[0])

				stddev = np.std(np.array(predictions), axis=0)
				print("hello2")
				print(stddev.shape)

				loss = np.sum(stddev ** 2) * -1

			return {'loss': loss, 'status': STATUS_OK, 'features': features}



		space = {
				 'k': hp.choice('k_choice',
								[
									(1.0),
									(hp.uniform('k_specified', 0, 1))
								]),
				 'accuracy': hp.choice('accuracy_choice',
								[
									(0.0),
									(hp.uniform('accuracy_specified', 0.5, 1))
								]),
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
				}

		trials = Trials()
		runs_per_dataset = 0
		i = 1
		while True:
			fmin(objective, space=space, algo=tpe.suggest, max_evals=i, trials=trials)
			i += 1

			#break, once convergence tolerance is reached and generate new dataset
			if trials.trials[-1]['result']['loss'] == 0 or i % 10 == 0:
				best_trial = trials.trials[-1]
				if i % 10 == 0:
					best_trial = trials.best_trial
				most_uncertain_f = best_trial['misc']['vals']
				#print(most_uncertain_f)

				min_accuracy = 0.0
				if most_uncertain_f['accuracy_choice'][0]:
					min_accuracy = most_uncertain_f['accuracy_specified'][0]
				min_fairness = 0.0
				if most_uncertain_f['fairness_choice'][0]:
					min_fairness = most_uncertain_f['fairness_specified'][0]
				min_robustness = 0.0
				if most_uncertain_f['robustness_choice'][0]:
					min_robustness = most_uncertain_f['robustness_specified'][0]
				max_number_features = X_train.shape[1]
				if most_uncertain_f['k_choice'][0]:
					max_number_features = most_uncertain_f['k_specified'][0]


				# Execute each search strategy with a given time limit (in parallel)
				# maybe run multiple times to smooth stochasticity

				model = LogisticRegression()
				if most_uncertain_f['privacy_choice'][0]:
					model = models.LogisticRegression(epsilon=most_uncertain_f['privacy_specified'][0])

				rankings = [variance, chi2_score_wo]  # simple rankings
				rankings.append(partial(model_score, estimator=ExtraTreesClassifier(n_estimators=1000)))  # accuracy ranking
				rankings.append(partial(robustness_score, model=model, scorer=auc_scorer))  # robustness ranking
				rankings.append(partial(fairness_score, estimator=ExtraTreesClassifier(n_estimators=1000),sensitive_ids=sensitive_ids))  # fairness ranking

				selected_rankings = rankings
				if type(ranking_id) != type(None):
					selected_rankings = [rankings[ranking_id]]


				result = strategy_method(X_train, X_test, y_train, y_test, names, sensitive_ids,
							 ranking_functions=selected_rankings,
							 clf=model,
							 min_accuracy=min_accuracy,
							 min_fairness=min_fairness,
							 min_robustness=min_robustness,
							 max_number_features=max_number_features,
							 max_search_time=time_limit,
							 cv_splitter=cv_splitter)

				# append ml data
				X_train_meta_classifier.append(best_trial['result']['features'])
				y_train_meta_classifier.append(result['time'])

				try:
					meta_classifier.fit(np.array(X_train_meta_classifier), y_train_meta_classifier)
				except:
					pass

				#pickle everything and store it
				one_big_object = {}
				one_big_object['features'] = X_train_meta_classifier
				one_big_object['best_strategy'] = y_train_meta_classifier

				pickle.dump(one_big_object, open('/tmp/regression_data_startegy_' + str(strategy_id) + '.pickle', 'wb'))

				trials = Trials()
				i = 1
				runs_per_dataset += 1
				break




strategy_id = 1
for r in range(5):
	configuration = {}
	configuration['ranking_functions'] = r
	configuration['main_strategy'] = weighted_ranking
	configuration['strategy_id'] = copy.deepcopy(strategy_id)
	mp_global.configurations.append(configuration)
	strategy_id +=1

main_strategies = [weighted_ranking, hyperparameter_optimization, evolution]

#run main strategies
for strategy in main_strategies:
	configuration = {}
	configuration['ranking_functions'] = None
	configuration['main_strategy'] = strategy
	configuration['strategy_id'] = copy.deepcopy(strategy_id)
	mp_global.configurations.append(configuration)
	strategy_id += 1


def my_function(config_id):
	conf = mp_global.configurations[config_id]
	run_strategy(conf['main_strategy'], conf['ranking_functions'], conf['strategy_id'])

with ProcessPool() as pool:
	future = pool.map(my_function, range(len(mp_global.configurations)))#, timeout=time_limit)

	iterator = future.result()
	while True:
		try:
			result = next(iterator)
		except StopIteration:
			break
		except TimeoutError as error:
			print("function took longer than %d seconds" % error.args[1])
		except ProcessExpired as error:
			print("%s. Exit code: %d" % (error, error.exitcode))
		except Exception as error:
			print("function raised %s" % error)
			print(error.traceback)  # Python's traceback of remote process

