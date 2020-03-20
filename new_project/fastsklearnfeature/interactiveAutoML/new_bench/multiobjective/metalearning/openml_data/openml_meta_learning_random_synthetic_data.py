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
import hyperopt.pyll.stochastic

import numpy as np
import copy
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
import random
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification



def generate_data(n_samples,n_features, configuration, test_samples = 1000):
	feature_distribution = np.array(
		[configuration['n_informative'], configuration['n_redundant'], configuration['n_repeated'],
		 configuration['n_useless']])
	feature_distribution /= np.sum(feature_distribution)
	feature_distribution *= n_features

	X, y = make_classification(n_samples=n_samples + test_samples,
							   n_features=n_features,
							   n_informative=int(feature_distribution[0]),
							   n_redundant=int(feature_distribution[1]),
							   n_repeated=int(feature_distribution[2]),
							   random_state=42,
							   n_clusters_per_class=configuration['n_clusters_per_class'])
	return X, y



def get_synthetic_data(n_samples,n_features, configuration, test_samples=1000):

	X, y = generate_data(n_samples,n_features, configuration, test_samples=test_samples)

	continuous_columns = list(range(X.shape[1]))


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_samples,
														random_state=42, stratify=y)



	my_transformers = []
	if len(continuous_columns) > 0:
		scale = ColumnTransformer([("scale", Pipeline(
			[('impute', SimpleImputer(missing_values=np.nan, strategy='mean')), ('scale', MinMaxScaler())]),
									continuous_columns)])
		my_transformers.append(("s", scale))

	pipeline = FeatureUnion(my_transformers)
	pipeline.fit(X_train)
	X_train = pipeline.transform(X_train)
	X_test = pipeline.transform(X_test)

	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)

	return X_train, X_test, y_train, y_test






current_run_time_id = time.time()

time_limit = 60 * 60 * 3
n_jobs = 20
number_of_runs = 1


ranking_scores_info = []


acc_value_list = []
fair_value_list = []
robust_value_list = []
success_value_list = []
runtime_value_list = []
evaluation_value_list = []
k_value_list = []

configuration_list = []
scaling_factor_list = []

cv_splitter = StratifiedKFold(5, random_state=42)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


### generate 100 configurations


space = {
		     ### constraint space
			 'k': hp.choice('k_choice',
							[
								(1.0),
								(hp.uniform('k_specified', 0, 1))
							]),
			 'accuracy': hp.uniform('accuracy_specified', 0.5, 1),
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
			 ### dataset space
		     'n_informative': hp.uniform('informative_specified', 0, 1),
			 'n_redundant': hp.uniform('redundant_specified', 0, 1),
		     'n_repeated': hp.uniform('repeated_specified', 0, 1),
		     'n_useless': hp.uniform('useless_specified', 0, 1),
		     'n_clusters_per_class': hp.randint('clusters_specified', 1,10),

			}

configurations = []
try:
	configurations = pickle.load(open(Config.get('data_path') + "/scaling_configurations_samples/scaling_configurations.pickle", "rb"))
except:
	while len(configurations) < 100:
		my_config = hyperopt.pyll.stochastic.sample(space)
		try:
			generate_data(100, 50, my_config, 0)
			configurations.append(my_config)
		except:
			continue


	pickle.dump(configurations, open(Config.get('data_path') + "/scaling_configurations_samples/scaling_configurations.pickle", 'wb'))


how_many_samples = int(input('enter number of samples please: '))

for number_samples in [how_many_samples]:#[100, 1000, 10000, 100000]:

	for config in configurations:
		X_train, X_test, y_train, y_test = get_synthetic_data(100, 50, config, test_samples=1000)

		mp_global.X_train = X_train
		mp_global.X_test = X_test
		mp_global.y_train = y_train
		mp_global.y_test = y_test
		mp_global.names = []
		mp_global.sensitive_ids = None
		mp_global.cv_splitter = cv_splitter


		min_accuracy = config['accuracy']
		min_fairness = 0.0
		min_robustness = config['robustness']
		max_number_features = config['k']

		max_search_time = time_limit

		model = LogisticRegression()
		if type(config['privacy']) != type(None):
			model = models.LogisticRegression(epsilon=config['privacy'])
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

		#pickle everything and store it
		one_big_object = {}
		one_big_object['ranking_scores_info'] = ranking_scores_info

		runtime_value_list.append(runtime_values)
		acc_value_list.append(accuracy_values)
		fair_value_list.append(fairness_values)
		robust_value_list.append(robustness_values)
		success_value_list.append(success_values)
		evaluation_value_list.append(evaluation_values)
		k_value_list.append(k_values)
		configuration_list.append(config)
		scaling_factor_list.append(number_samples)

		one_big_object['times_value'] = runtime_value_list
		one_big_object['k_value'] = k_value_list
		one_big_object['acc_value'] = acc_value_list
		one_big_object['fair_value'] = fair_value_list
		one_big_object['robust_value'] = robust_value_list
		one_big_object['success_value'] = success_value_list
		one_big_object['evaluation_value'] = evaluation_value_list
		one_big_object['config_value'] = configuration_list
		one_big_object['scaling_samples_value'] = scaling_factor_list

		pickle.dump(one_big_object, open('/tmp/metalearning_data' + str(current_run_time_id) + 'scaling_factor_' + str(number_samples) +  '.pickle', 'wb'))


