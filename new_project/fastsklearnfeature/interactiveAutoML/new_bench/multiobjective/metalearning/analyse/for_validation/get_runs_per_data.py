import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_recall
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import time_score2
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_avg_runtime
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_optimum_avg_runtime

from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
import copy
import glob


mappnames = {1:'TPE(Variance)',
			 2: 'TPE($\chi^2$)',
			 3:'TPE(FCBF)',
			 4: 'TPE(Fisher)',
			 5: 'TPE(MIM)',
			 6: 'TPE(MCFS)',
			 7: 'TPE(ReliefF)',
			 8: 'TPE(NR)',
             9: 'SA(NR)',
			 10: 'NSGA-II(NR)',
			 11: 'ES(NR)',
			 12: 'SFS(NR)',
			 13: 'SBS(NR)',
			 14: 'SFFS(NR)',
			 15: 'SBFS(NR)',
			 16: 'RFE(LR)',
			 17: 'Complete Set'
			 }

names = ['accuracy',
	 'fairness',
	 'k_rel',
	 'k',
	 'robustness',
	 'privacy',
	 'search_time',
	 'cv_acc - acc',
	 'cv_fair - fair',
	 'cv_k - k rel',
	 'cv_k - k',
	 'cv_robust - robust',
     'cv time',
	 'rows',
	 'columns']

def print_constraints_2(features):


	my_str = ''
	for i in range(len(names)):
		my_str += names[i] + ': ' + str(features[i]) + ' '
	print(my_str)




experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")

print(experiment_folders)


dataset = {}
dataset['best_strategy'] = []
dataset['validation_satisfied'] = []


dataset['success_value'] = []
dataset['success_value_validation'] = []
dataset['times_value'] = []
dataset['max_search_time'] = []
dataset['dataset_id'] = []

dataset['distance_to_test_constraint'] = []


def load_pickle(fname):
	data = []
	with open(fname, "rb") as f:
		while True:
			try:
				data.append(pickle.load(f))
			except EOFError:
				break
	return data


def is_successfull_validation_and_test(exp_results):
	return len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1]['success_test'] == True #also on test satisfied

def is_successfull_validation(exp_results):
	return len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[-1]  # constraints were satisfied on validation set



def distance_to_constraints_on_test(exp_results, best_run, info_dict):
	min_fairness = info_dict['constraint_set_list']['fairness']
	min_accuracy = info_dict['constraint_set_list']['accuracy']
	min_robustness = info_dict['constraint_set_list']['robustness']
	max_number_features = info_dict['constraint_set_list']['k']

	if type(best_run) != type(None):
		test_fair = exp_results[best_run]['test_fair']
		test_acc = exp_results[best_run]['test_acc']
		test_robust = exp_results[best_run]['test_robust']
		test_number_features = exp_results[best_run]['cv_number_features']
	else:
		test_fair = 0.0
		test_acc = 0.0
		test_robust = 0.0
		test_number_features = 1.0



	loss = 0.0
	if min_fairness > 0.0 and test_fair < min_fairness:
		loss += (min_fairness - test_fair) ** 2
	if min_accuracy > 0.0 and test_acc < min_accuracy:
		loss += (min_accuracy - test_acc) ** 2
	if min_robustness > 0.0 and test_robust < min_robustness:
		loss += (min_robustness - test_robust) ** 2
	if max_number_features < 1.0 and test_number_features > max_number_features:
		loss += (test_number_features - max_number_features) ** 2

	return loss

def distance_to_constraints_on_validation(exp_results, best_run, info_dict):
	min_fairness = info_dict['constraint_set_list']['fairness']
	min_accuracy = info_dict['constraint_set_list']['accuracy']
	min_robustness = info_dict['constraint_set_list']['robustness']
	max_number_features = info_dict['constraint_set_list']['k']

	if type(best_run) != type(None):
		test_fair = exp_results[best_run]['cv_fair']
		test_acc = exp_results[best_run]['cv_acc']
		test_robust = exp_results[best_run]['cv_robust']
		test_number_features = exp_results[best_run]['cv_number_features']
	else:
		test_fair = 0.0
		test_acc = 0.0
		test_robust = 0.0
		test_number_features = 1.0



	loss = 0.0
	if min_fairness > 0.0 and test_fair < min_fairness:
		loss += (min_fairness - test_fair) ** 2
	if min_accuracy > 0.0 and test_acc < min_accuracy:
		loss += (min_accuracy - test_acc) ** 2
	if min_robustness > 0.0 and test_robust < min_robustness:
		loss += (min_robustness - test_robust) ** 2
	if max_number_features < 1.0 and test_number_features > max_number_features:
		loss += (test_number_features - max_number_features) ** 2

	return loss


strategy_distance_test = {}
strategy_distance_validation = {}
for s in range(1, len(mappnames) + 1):
	strategy_distance_test[s] = []
	strategy_distance_validation[s] = []


number_ml_scenarios = 1200
run_count = 0
for efolder in experiment_folders:
	run_folders = sorted(glob.glob(efolder + "*/"))
	for rfolder in run_folders:
		try:
			info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))
			run_strategies_success_test = {}
			run_strategies_times = {}
			run_strategies_success_validation = {}

			validation_satisfied_by_any_strategy = False

			min_time = np.inf
			best_strategy = 0
			for s in range(1, len(mappnames) + 1):
				exp_results = []
				try:
					exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
				except:
					pass
				if is_successfull_validation_and_test(exp_results):
					runtime = exp_results[-1]['final_time']
					if runtime < min_time:
						min_time = runtime
						best_strategy = s

					run_strategies_success_test[s] = True
					run_strategies_times[s] = runtime
				else:
					run_strategies_success_test[s] = False
					#run_strategies_times[s] = runtime

				run_strategies_success_validation[s] = is_successfull_validation(exp_results)
				if run_strategies_success_validation[s]:
					validation_satisfied_by_any_strategy = True

			dataset['success_value'].append(run_strategies_success_test)
			dataset['success_value_validation'].append(run_strategies_success_validation)
			dataset['best_strategy'].append(best_strategy)
			dataset['times_value'].append(run_strategies_times)
			dataset['validation_satisfied'].append(validation_satisfied_by_any_strategy)

			dataset['max_search_time'].append(info_dict['constraint_set_list']['search_time'])
			dataset['dataset_id'].append(info_dict['dataset_id'])

			run_count += 1
		except FileNotFoundError:
			pass

		if run_count == number_ml_scenarios:
			break
	if run_count == number_ml_scenarios:
		break

map_dataset2name = {}
map_dataset2name['31'] = 'German Credit'
map_dataset2name['802'] = 'Primary Biliary Cirrhosis'
map_dataset2name['1590'] = 'Adult'
map_dataset2name['1461'] = 'Bank Marketing'
map_dataset2name['42193'] = 'COMPAS'
map_dataset2name['1480'] = 'Indian Liver Patient'
#map_dataset2name['804'] = 'hutsof99_logis'
map_dataset2name['42178'] = 'Telco Customer Churn'
map_dataset2name['981'] = 'KDD Internet Usage'
map_dataset2name['40536'] = 'Speed Dating'
map_dataset2name['40945'] = 'Titanic'
map_dataset2name['451'] = 'Irish Educational Transitions'
#map_dataset2name['945'] = 'Kidney'
map_dataset2name['446'] = 'Leptograpsus crabs'
map_dataset2name['1017'] = 'Arrhythmia'
map_dataset2name['957'] = 'Brazil Tourism'
map_dataset2name['41430'] = 'Diabetic Mellitus'
map_dataset2name['1240'] = 'AirlinesCodrnaAdult'
map_dataset2name['1018'] = 'IPUMS Census'
#map_dataset2name['55'] = 'Hepatitis'
map_dataset2name['38'] = 'Thyroid Disease'
map_dataset2name['1003'] = 'Primary Tumor'
map_dataset2name['934'] ='Social Mobility'


datasets_ids = np.unique(dataset['dataset_id'])

for d in range(len(datasets_ids)):
	print(map_dataset2name[datasets_ids[d]] + ' ' + str(np.sum(np.array(dataset['dataset_id']) == datasets_ids[d])))