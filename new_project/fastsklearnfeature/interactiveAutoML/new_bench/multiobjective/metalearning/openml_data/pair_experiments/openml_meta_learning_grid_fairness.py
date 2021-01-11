import copy
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_mcfs
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
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
import os

#static constraints: fairness, number of features (absolute and relative), robustness, privacy, accuracy

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_fair_data1_validation
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score


def load_pickle(fname):
	data = []
	with open(fname, "rb") as f:
		while True:
			try:
				data.append(pickle.load(f))
			except EOFError:
				break
	return data


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

number_runs = 1

cv_splitter = StratifiedKFold(5, random_state=42)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

mp_global.X_train = []
mp_global.X_validation = []
mp_global.X_train_val = []
mp_global.X_test = []
mp_global.y_train = []
mp_global.y_validation = []
mp_global.y_train_val = []
mp_global.y_test = []
mp_global.names = []
mp_global.sensitive_ids = []
mp_global.cv_splitter = []


for nruns in range(5):
	X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, key, sensitive_attribute_id = get_fair_data1_validation(
		dataset_key='1590', random_number=42 + nruns)

	mp_global.X_train.append(X_train)
	mp_global.X_validation.append(X_validation)
	mp_global.X_train_val.append(X_train_val)
	mp_global.X_test.append(X_test)
	mp_global.y_train.append(y_train)
	mp_global.y_validation.append(y_validation)
	mp_global.y_train_val.append(y_train_val)
	mp_global.y_test.append(y_test)
	mp_global.names.append(names)
	mp_global.sensitive_ids.append(sensitive_ids)
	mp_global.cv_splitter.append(cv_splitter)

	mp_global.accuracy_scorer = make_scorer(f1_score)
	mp_global.avoid_robustness = False

runs_per_dataset = 0
l_acc = 0.40
u_acc = 0.70
l_fair = 0.80
u_fair = 0.91


print(list(np.arange(l_acc, u_acc, (u_acc - l_acc) / 10.0)))

results_heatmap = {}
for min_accuracy in [0.5, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68]:
	for min_fairness in [0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]:

		success_per_strategy = np.zeros(18)
		time_per_strategy = np.zeros(18)
		for nruns_global in range(5):

			min_robustness = 0.0
			max_number_features = 1.0
			max_search_time = 20 * 60
			privacy = None

			# Execute each search strategy with a given time limit (in parallel)
			# maybe run multiple times to smooth stochasticity

			model = LogisticRegression(class_weight='balanced')
			if type(privacy) != type(None):
				model = models.LogisticRegression(epsilon=privacy, class_weight='balanced')
			mp_global.clf = model
			mp_global.model_hyperparameters = {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}

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
							   recursive_feature_elimination,
							   fullfeatures]

			#run main strategies
			for strategy in main_strategies:
				for run in[0]:
						configuration = {}
						configuration['ranking_functions'] = rankings
						configuration['run_id'] = copy.deepcopy(run)
						configuration['main_strategy'] = strategy
						configuration['strategy_id'] = copy.deepcopy(strategy_id)
						mp_global.configurations.append(configuration)
				strategy_id += 1

			def my_function(config_id):
				run_i = nruns_global
				conf = mp_global.configurations[config_id]
				log_file = '/tmp/experiment' + str(current_run_time_id) + '_run_' + str(run_i) + '_strategy' + str(conf['strategy_id']) + '.pickle'

				result = conf['main_strategy'](mp_global.X_train[run_i],
											   mp_global.X_validation[run_i],
											   mp_global.X_train_val[run_i],
											   mp_global.X_test[run_i],
											   mp_global.y_train[run_i],
											   mp_global.y_validation[run_i],
											   mp_global.y_train_val[run_i],
											   mp_global.y_test[run_i],
											   mp_global.names[run_i],
											   mp_global.sensitive_ids[run_i],
											   ranking_functions=conf['ranking_functions'],
											   clf=mp_global.clf,
											   min_accuracy=mp_global.min_accuracy,
											   min_fairness=mp_global.min_fairness,
											   min_robustness=mp_global.min_robustness,
											   max_number_features=mp_global.max_number_features,
											   max_search_time=mp_global.max_search_time,
											   log_file=log_file,
											   accuracy_scorer=mp_global.accuracy_scorer,
											   model_hyperparameters=mp_global.model_hyperparameters)


				result['strategy_id'] = conf['strategy_id']
				if result['success']:
					exp_results = load_pickle(log_file)
					result['time'] = exp_results[-1]['final_time']
				os.remove(log_file)

				return result



			check_strategies = np.zeros(strategy_id)
			with ProcessPool(max_workers=17) as pool:
				future = pool.map(my_function, range(len(mp_global.configurations)), timeout=max_search_time)

				iterator = future.result()
				while True:
					try:
						result = next(iterator)
						if result['success'] == True:
							try:
								success_per_strategy[result['strategy_id']] += 1
								time_per_strategy[result['strategy_id']] += result['time']
								pool.stop()
								pool.join(timeout=0)
								break
							except:
								print("fail strategy Id: " + str(result['strategy_id']))
					except StopIteration:
						break
					except TimeoutError as error:
						print("function took longer than %d seconds" % error.args[1])
					except ProcessExpired as error:
						print("%s. Exit code: %d" % (error, error.exitcode))
					except Exception as error:
						print("function raised %s" % error)
						print(error.traceback)  # Python's traceback of remote process

		# do logging
		print('my heat map is here: ' + str(results_heatmap))
		with open('/tmp/current_heat_map_fair_acc.txt', 'w+') as f:
			f.write(str(results_heatmap) + ' current position: fair: ' + str(min_fairness) + ' acc: ' + str(
				min_accuracy))

		if np.sum(success_per_strategy) == 0:
			break
		else:
			fastest_strategy_id = np.argmax(success_per_strategy)

			results_heatmap[(min_accuracy, min_fairness)] = (time_per_strategy[fastest_strategy_id] / float(success_per_strategy[fastest_strategy_id]), fastest_strategy_id)

			with open('/tmp/current_heat_map_fair_acc.pickle', 'wb+') as f_log:
				pickle.dump(results_heatmap, f_log, protocol=pickle.HIGHEST_PROTOCOL)


print('my heat map is here: ' + str(results_heatmap))


