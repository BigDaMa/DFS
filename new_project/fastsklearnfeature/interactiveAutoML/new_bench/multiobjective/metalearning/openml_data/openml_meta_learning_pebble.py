import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time

from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.metrics import loss_sensitivity
from art.metrics import empirical_robustness
from sklearn.pipeline import FeatureUnion


from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from art.classifiers import XGBoostClassifier, LightGBMClassifier, SklearnClassifier
from art.attacks import HopSkipJump


from fastsklearnfeature.interactiveAutoML.feature_selection.RunAllKBestSelection import RunAllKBestSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fairness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import robustness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import unit_test_score

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test

from fastsklearnfeature.configuration.Config import Config

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif


from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score

from sklearn.ensemble import RandomForestRegressor
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.multiprocessing_global as mp_global

import sklearn

import diffprivlib.models as models
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import hyperparameter_optimization
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.evolution import evolution

#static constraints: fairness, number of features (absolute and relative), robustness, privacy, accuracy

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_data_openml
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

#load list of viable datasets
data_infos = pickle.load(open(Config.get('data_path') + '/openml_data/fitting_datasets.pickle', 'rb'))

time_limit = 60 * 20
n_jobs = 20
number_of_runs = 2

meta_classifier = RandomForestClassifier(n_estimators=1000)
X_train_meta_classifier = []
y_train_meta_classifier = []


acc_value_list = []
fair_value_list = []
robust_value_list = []
success_value_list = []
runtime_value_list = []

cv_splitter = StratifiedKFold(5, random_state=42)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

while True:
	X_train, X_test, y_train, y_test, names, sensitive_ids = get_data_openml(data_infos)

	#run on tiny sample
	X_train_tiny, _, y_train_tiny, _ = train_test_split(X_train, y_train, train_size=100, random_state=42, stratify=y_train)

	fair_train_tiny = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train_tiny[:, sensitive_ids[0]])

	mp_global.X_train = X_train
	mp_global.X_test = X_test
	mp_global.y_train = y_train
	mp_global.y_test = y_test
	mp_global.names = names
	mp_global.sensitive_ids = sensitive_ids
	mp_global.max_search_time = time_limit
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
			try:
				proba_predictions = meta_classifier.predict_proba([features])[0]
				proba_predictions = np.sort(proba_predictions)

				print("predictions: " + str(proba_predictions))

				uncertainty = 1 - (proba_predictions[-1] - proba_predictions[-2])
				loss = -1 * uncertainty  # we want to maximize uncertainty
			except:
				pass

			return {'loss': loss, 'status': STATUS_OK, 'features': features}
		except:
			return {'loss': np.inf, 'status': STATUS_OK}



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

		if trials.trials[-1]['result']['loss'] == np.inf:
			break

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
			mp_global.clf = model

			#define rankings
			rankings = [variance, chi2_score_wo] #simple rankings
			rankings.append(partial(model_score, estimator=ExtraTreesClassifier(n_estimators=1000))) #accuracy ranking
			rankings.append(partial(robustness_score, model=model, scorer=auc_scorer)) #robustness ranking
			rankings.append(partial(fairness_score, estimator=ExtraTreesClassifier(n_estimators=1000), sensitive_ids=sensitive_ids)) #fairness ranking


			mp_global.min_accuracy = min_accuracy
			mp_global.min_fairness = min_fairness
			mp_global.min_robustness = min_robustness
			mp_global.max_number_features = max_number_features

			mp_global.configurations = []
			#add single rankings
			strategy_id = 1
			for r in range(len(rankings)):
				for run in range(number_of_runs):
					configuration = {}
					configuration['ranking_functions'] = [rankings[r]]
					configuration['run_id'] = run
					configuration['main_strategy'] = weighted_ranking
					configuration['strategy_id'] = copy.deepcopy(strategy_id)
					mp_global.configurations.append(configuration)
				strategy_id +=1

			main_strategies = [weighted_ranking, hyperparameter_optimization, evolution]

			#run main strategies
			for strategy in main_strategies:
				for run in range(number_of_runs):
						configuration = {}
						configuration['ranking_functions'] = rankings
						configuration['run_id'] = run
						configuration['main_strategy'] = strategy
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
			with ProcessPool() as pool:
				future = pool.map(my_function, range(len(mp_global.configurations)), timeout=time_limit)

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
					else:
						accuracy_values[results[r]['strategy_id']].append(results[r]['cv_acc'])
						fairness_values[results[r]['strategy_id']].append(results[r]['cv_fair'])
						robustness_values[results[r]['strategy_id']].append(results[r]['cv_robust'])
						k_values[results[r]['strategy_id']].append(results[r]['cv_number_features'])
						success_values[results[r]['strategy_id']].append(results[r]['success'])
						runtime_values[results[r]['strategy_id']].append(results[r]['time'])
				if results[r]['success']:
					success[results[r]['strategy_id']] = True


			for strategy_i in range(1, strategy_id):
				number_successes = 0
				if strategy_i in success_values:
					number_successes += np.sum(success_values[strategy_i])
				runtimes[strategy_i] += (number_of_runs - number_successes) * time_limit

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

			try:
				meta_classifier.fit(np.array(X_train_meta_classifier), y_train_meta_classifier)
			except:
				pass

			#pickle everything and store it
			one_big_object = {}
			one_big_object['features'] = X_train_meta_classifier
			one_big_object['best_strategy'] = y_train_meta_classifier

			runtime_value_list.append(runtime_values)
			acc_value_list.append(accuracy_values)
			fair_value_list.append(fairness_values)
			robust_value_list.append(robustness_values)
			success_value_list.append(success_values)

			one_big_object['times_value'] = runtime_value_list
			one_big_object['acc_value'] = acc_value_list
			one_big_object['fair_value'] = fair_value_list
			one_big_object['robust_value'] = robust_value_list
			one_big_object['success_value'] = success_value_list

			pickle.dump(one_big_object, open('/tmp/metalearning_data.pickle', 'wb'))

			trials = Trials()
			i = 1
			runs_per_dataset += 1
			break


