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

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_fair_data
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

#load list of viable datasets
data_infos = pickle.load(open(Config.get('data_path') + '/openml_data/fitting_datasets.pickle', 'rb'))

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

while True:
	X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id = get_fair_data(dataset_key='1590')

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


	results_heatmap = {}
	for min_accuracy in np.arange(l_acc, u_acc, (u_acc - l_acc) / 10.0):
		for min_fairness in np.arange(l_fair, u_fair, (u_fair - l_fair) / 10.0):
			i += 1

			min_robustness = 0.0
			max_number_features = 1.0
			max_search_time = 10 * 60
			privacy = None

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
						if result['success'] == True:
							results_heatmap [(min_accuracy, min_fairness)] = (result['runtime'], result['strategy_id'])
							print(results_heatmap)
							pool.stop()
							pool.join(timeout=0)
							break
					except StopIteration:
						break
					except TimeoutError as error:
						print("function took longer than %d seconds" % error.args[1])
					except ProcessExpired as error:
						print("%s. Exit code: %d" % (error, error.exitcode))
					except Exception as error:
						print("function raised %s" % error)
						print(error.traceback)  # Python's traceback of remote process



