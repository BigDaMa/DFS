from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import time

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_mcfs
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from skrebate import ReliefF
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.multiprocessing_global as mp_global
import diffprivlib.models as models

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.fullfeatures import fullfeatures
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
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import hyperopt.pyll.stochastic

import numpy as np
import copy
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
import os

from sklearn.metrics import f1_score

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.private_models.randomforest.PrivateRandomForrest import PrivateRandomForest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

how_many_samples = int(input('enter number of samples please: '))

def my_function(config_id):
	conf = mp_global.configurations[config_id]
	result = conf['main_strategy'](mp_global.X_train,
								   mp_global.X_validation,
								   mp_global.X_train_val,
								   mp_global.X_test,
								   mp_global.y_train,
								   mp_global.y_validation,
								   mp_global.y_train_val,
								   mp_global.y_test,
								   mp_global.names,
								   mp_global.sensitive_ids,
								   ranking_functions=conf['ranking_functions'],
								   clf=mp_global.clf,
								   min_accuracy=mp_global.min_accuracy,
								   min_fairness=mp_global.min_fairness,
								   min_robustness=mp_global.min_robustness,
								   max_number_features=mp_global.max_number_features,
								   max_search_time=mp_global.max_search_time,
								   log_file='/tmp/experiment_features' + str(how_many_samples) + '/run' + str(run_counter) + '/strategy' + str(conf['strategy_id']) + '.pickle',
								   accuracy_scorer=mp_global.accuracy_scorer)
	result['strategy_id'] = conf['strategy_id']
	return result



def generate_data(n_samples,n_features, configuration, test_samples = 1000):
	feature_distribution = np.array(
		[configuration['n_informative'], configuration['n_redundant'], configuration['n_repeated'],
		 configuration['n_useless']])
	feature_distribution /= np.sum(feature_distribution)
	feature_distribution *= n_features

	X, y = make_classification(n_samples=n_samples + test_samples*2,
							   n_features=n_features,
							   n_informative=int(feature_distribution[0]),
							   n_redundant=int(feature_distribution[1]),
							   n_repeated=int(feature_distribution[2]),
							   random_state=42,
							   n_clusters_per_class=configuration['n_clusters_per_class'])
	return X, y

def generate_data_one(n_samples,n_features, test_samples = 1000):
	X, y = make_classification(n_samples=n_samples + test_samples*2,
							   n_features=n_features,
							   n_informative=5,
							   n_redundant=15,
							   n_repeated=0,
							   random_state=42,
							   n_clusters_per_class=16,
							   n_classes=2)
	return X, y



def get_synthetic_data(n_samples,n_features, configuration, test_samples=1000, random_state=42):

	X, y = generate_data_one(n_samples, n_features, test_samples=test_samples)

	continuous_columns = list(range(X.shape[1]))


	X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_samples,random_state=random_state, stratify=y)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_samples, random_state=random_state, stratify=y)



	my_transformers = []
	if len(continuous_columns) > 0:
		scale = ColumnTransformer([("scale", Pipeline(
			[('impute', SimpleImputer(missing_values=np.nan, strategy='mean')), ('scale', MinMaxScaler())]),
									continuous_columns)])
		my_transformers.append(("s", scale))

	pipeline = FeatureUnion(my_transformers)
	pipeline.fit(X_train)
	X_train = pipeline.transform(X_train)
	X_val = pipeline.transform(X_val)
	X_test = pipeline.transform(X_test)

	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.fit_transform(y_train)
	y_val = le.transform(y_val)
	y_test = le.transform(y_test)

	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.append(y_train, y_val)

	return X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test






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
			 'model':hp.choice('model_choice',
							[
								'Logistic Regression',
								'Gaussian Naive Bayes',
								'Decision Tree' #, 'Random Forest'
							]),
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
								(hp.uniform('robustness_specified', 0.8, 1))
							]),
			 #TODO: set these to default
			 ### dataset space
		     'n_informative': hp.uniform('informative_specified', 0, 1),
			 'n_redundant': hp.uniform('redundant_specified', 0, 1),
		     'n_repeated': hp.uniform('repeated_specified', 0, 1),
		     'n_useless': hp.uniform('useless_specified', 0, 1),
		     'n_clusters_per_class': hp.randint('clusters_specified', 1,10),

			}

configurations = []
try:
	configurations = pickle.load(open(Config.get('data_path') + "/scaling_configurations_samples/scaling_configurations_modelsf.pickle", "rb"))
except:
	while len(configurations) < 100:
		my_config = hyperopt.pyll.stochastic.sample(space)
		try:
			generate_data(100, 50, my_config, 0)
			configurations.append(my_config)
		except:
			continue


	pickle.dump(configurations, open(Config.get('data_path') + "/scaling_configurations_samples/scaling_configurations_modelsf.pickle", 'wb'))


for number_features in [how_many_samples]:

	os.mkdir('/tmp/experiment_features' + str(number_features))

	run_counter = 0
	for config in configurations:

		os.mkdir('/tmp/experiment_features' + str(number_features) + '/run' + str(run_counter))

		X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test = get_synthetic_data(10000, number_features, config, test_samples=1000)

		mp_global.X_train = X_train
		mp_global.X_test = X_test
		mp_global.y_train = y_train
		mp_global.y_test = y_test
		mp_global.names = []
		mp_global.sensitive_ids = None

		mp_global.cv_splitter = StratifiedKFold(5, random_state=42)
		mp_global.accuracy_scorer = make_scorer(f1_score)
		mp_global.avoid_robustness = False


		mp_global.X_validation = X_val
		mp_global.X_train_val = X_train_val
		mp_global.y_validation = y_val
		mp_global.y_train_val = y_train_val


		min_accuracy = config['accuracy']
		min_fairness = 0.0
		min_robustness = config['robustness']
		max_number_features = config['k']

		max_search_time = time_limit

		model = None
		if config['model'] == 'Logistic Regression':
			model = LogisticRegression(class_weight='balanced')
			if type(config['privacy']) != type(None):
				model = models.LogisticRegression(epsilon=config['privacy'],
												  class_weight='balanced')
		elif config['model'] == 'Gaussian Naive Bayes':
			model = GaussianNB()
			if type(config['privacy']) != type(None):
				model = models.GaussianNB(epsilon=config['privacy'])
		elif config['model'] == 'Decision Tree':
			model = DecisionTreeClassifier(class_weight='balanced')
			if type(config['privacy']) != type(None):
				model = PrivateRandomForest(n_estimators=1, epsilon=config['privacy'])

		mp_global.clf = model
		# define rankings
		rankings = [variance,
					chi2_score_wo,
					fcbf,
					my_fisher_score,
					mutual_info_classif,
					my_mcfs]
		# rankings.append(partial(model_score, estimator=ExtraTreesClassifier(n_estimators=1000))) #accuracy ranking
		# rankings.append(partial(robustness_score, model=model, scorer=auc_scorer)) #robustness ranking
		# rankings.append(partial(fairness_score, estimator=ExtraTreesClassifier(n_estimators=1000), sensitive_ids=sensitive_ids)) #fairness ranking
		rankings.append(partial(model_score, estimator=ReliefF(n_neighbors=10)))  # relieff

		mp_global.min_accuracy = min_accuracy
		mp_global.min_fairness = min_fairness
		mp_global.min_robustness = min_robustness
		mp_global.max_number_features = max_number_features
		mp_global.max_search_time = max_search_time

		mp_global.configurations = []
		# add single rankings
		strategy_id = 1
		for r in range(len(rankings)):
			for run in range(number_of_runs):
				configuration = {}
				configuration['ranking_functions'] = copy.deepcopy([rankings[r]])
				configuration['run_id'] = copy.deepcopy(run)
				configuration['main_strategy'] = copy.deepcopy(weighted_ranking)
				configuration['strategy_id'] = copy.deepcopy(strategy_id)
				mp_global.configurations.append(configuration)
			strategy_id += 1

		main_strategies = [TPE,
						   simulated_annealing,
						   evolution,
						   exhaustive,
						   forward_selection,
						   backward_selection,
						   forward_floating_selection,
						   backward_floating_selection,
						   recursive_feature_elimination,
						   fullfeatures]

		# run main strategies
		for strategy in main_strategies:
			for run in range(number_of_runs):
				configuration = {}
				configuration['ranking_functions'] = []
				configuration['run_id'] = copy.deepcopy(run)
				configuration['main_strategy'] = copy.deepcopy(strategy)
				configuration['strategy_id'] = copy.deepcopy(strategy_id)
				mp_global.configurations.append(configuration)
			strategy_id += 1

		with ProcessPool(max_workers=17) as pool:
			future = pool.map(my_function, range(len(mp_global.configurations)), timeout=max_search_time)

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
			# print(error.traceback)  # Python's traceback of remote process

		# pickle everything and store it
		one_big_object = {}
		one_big_object['dataset_id'] = 'Madelon'
		one_big_object['constraint_set_list'] = config
		one_big_object['number_instances'] = 10000
		one_big_object['number_features'] = number_features

		with open('/tmp/experiment_features' + str(number_features) + '/run' + str(run_counter) + '/run_info.pickle','wb') as f_log:
			pickle.dump(one_big_object, f_log)

		run_counter += 1
