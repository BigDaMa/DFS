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

from sklearn.datasets import make_classification

from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
import random
from sklearn.impute import SimpleImputer

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
dataset_sensitive_attribute_list = []

cv_splitter = StratifiedKFold(5, random_state=42)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

def get_synthetic_data(number_observations, number_features):
	X, Y = make_classification(n_samples=number_observations, n_features=number_features, n_classes=2)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5,
														random_state=42, stratify=Y)

	continuous_columns = list(range(number_features))

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

	return X_train, X_test, y_train, y_test, [], None, None


save_results = {}

for number_observations in [200, 1000,2000,20000,200000,2000000]:
	X_train, X_test, y_train, y_test, names, sensitive_ids, sensitive_attribute_id = get_synthetic_data(number_observations, 20)

	mp_global.X_train = X_train
	mp_global.X_test = X_test
	mp_global.y_train = y_train
	mp_global.y_test = y_test
	mp_global.names = names
	mp_global.sensitive_ids = sensitive_ids
	mp_global.cv_splitter = cv_splitter

	runs_per_dataset = 0
	i = 1
	l_acc = 0.7
	u_acc = 0.91
	l_fair = 0.84
	u_fair = 1.0


	start_features = 2.0 / X_train.shape[1]

	print(np.arange(start_features, 1.0 + (1.0 - start_features) / 10.0, (1.0 - start_features) / 10.0))

	results_heatmap = {}
	i += 1

	min_accuracy = 1.0
	max_number_features = 1.0
	min_robustness = 1.0
	max_search_time = 20 * 60
	privacy = None
	min_fairness = 0.0

	# Execute each search strategy with a given time limit (in parallel)
	# maybe run multiple times to smooth stochasticity

	model = LogisticRegression()
	if type(privacy) != type(None):
		model = models.LogisticRegression(epsilon=privacy)
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
			configuration['ranking_functions'] = [rankings[r]]
			configuration['run_id'] = copy.deepcopy(run)
			configuration['main_strategy'] = weighted_ranking
			configuration['strategy_id'] = copy.deepcopy(strategy_id)
			mp_global.configurations.append(configuration)
		strategy_id +=1

	'''
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
				configuration['ranking_functions'] = rankings
				configuration['run_id'] = copy.deepcopy(run)
				configuration['main_strategy'] = strategy
				configuration['strategy_id'] = copy.deepcopy(strategy_id)
				mp_global.configurations.append(configuration)
		strategy_id += 1
	'''

	config_id = 0


	for config_id in range(len(mp_global.configurations)):
		ranking_times_all = []
		run_times_all_all = []
		for check_run in range(10):
			conf = mp_global.configurations[config_id]
			result = conf['main_strategy'](mp_global.X_train, mp_global.X_test, mp_global.y_train, mp_global.y_test, mp_global.names, mp_global.sensitive_ids,
						 ranking_functions=conf['ranking_functions'],
						 clf=mp_global.clf,
						 min_accuracy=mp_global.min_accuracy,
						 min_fairness=mp_global.min_fairness,
						 min_robustness=mp_global.min_robustness,
						 max_number_features=mp_global.max_number_features,
						 max_search_time=np.inf,
						 cv_splitter=mp_global.cv_splitter,
						 max_number_evals=10)
			result['strategy_id'] = conf['strategy_id']
			ranking_times_all.append(result['ranking_time'])
			run_times_all_all.extend(result['feature_selection_times'])

		print("Strategy: " + str(result['strategy_id']))
		print("average ranking time: " + str(np.mean(ranking_times_all)) + ' std: ' + str(np.std(ranking_times_all)))
		print("average feature selection without ranking - time: " + str(np.mean(run_times_all_all)) + ' std: ' + str(np.std(run_times_all_all)))
		print("\n")

		save_results[(number_observations, result['strategy_id'], 'ranking_avg')] = np.mean(ranking_times_all)
		save_results[(number_observations, result['strategy_id'], 'ranking_std')] = np.std(ranking_times_all)
		save_results[(number_observations, result['strategy_id'], 'selection_avg')] = np.mean(run_times_all_all)
		save_results[(number_observations, result['strategy_id'], 'selection_std')] = np.std(run_times_all_all)

		print(save_results)






