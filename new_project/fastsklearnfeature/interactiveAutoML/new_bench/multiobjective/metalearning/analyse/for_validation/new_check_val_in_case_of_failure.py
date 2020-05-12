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

finished_validation = {}
finished_test = {}

for s in range(1, len(mappnames) + 1):
	strategy_distance_test[s] = []
	strategy_distance_validation[s] = []

	finished_validation[s] = []
	finished_test[s] = []


oracle_distance_validation = []
oracle_distance_test = []
for efolder in experiment_folders:
	run_folders = glob.glob(efolder + "*/")
	for rfolder in run_folders:
		try:
			info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))

			distance_of_this_run_test = []
			distance_of_this_run_validation = []
			for s in range(1, len(mappnames) + 1):
				exp_results = []
				try:
					exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
				except:
					pass

				min_loss = np.inf
				best_run = None

				for min_r in range(len(exp_results)):
					if 'loss' in exp_results[min_r] and exp_results[min_r]['loss'] < min_loss:
						min_loss = exp_results[min_r]['loss']
						best_run = min_r

				my_dist = distance_to_constraints_on_test(exp_results, best_run, info_dict)
				if not is_successfull_validation_and_test(exp_results):
					strategy_distance_test[s].append(my_dist)
					distance_of_this_run_test.append(my_dist)

					finished_validation[s].append(len(exp_results) > 0 and 'Finished' in exp_results[-1] and exp_results[-1]['Finished'])

				#now distance to validation
				if not is_successfull_validation(exp_results):
					my_dist_validation = distance_to_constraints_on_validation(exp_results, best_run, info_dict)
					strategy_distance_validation[s].append(my_dist_validation)
					distance_of_this_run_validation.append(my_dist_validation)

					finished_test[s].append(
						len(exp_results) > 0 and 'Finished' in exp_results[-1] and exp_results[-1]['Finished'])

		except FileNotFoundError:
			pass


run_count = 0
for efolder in experiment_folders:
	run_folders = glob.glob(efolder + "*/")
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

					for exp_run in exp_results:
						if 'Finished' in exp_run and exp_run['Finished']:
							run_strategies_times[s] = exp_run['final_time']
							break
					if not s in run_strategies_times:
						run_strategies_times[s] = info_dict['constraint_set_list']['search_time']

				run_strategies_success_validation[s] = is_successfull_validation(exp_results)
				if run_strategies_success_validation[s]:
					validation_satisfied_by_any_strategy = True

			dataset['success_value'].append(run_strategies_success_test)
			dataset['success_value_validation'].append(run_strategies_success_validation)
			dataset['best_strategy'].append(best_strategy)
			dataset['times_value'].append(run_strategies_times)
			dataset['validation_satisfied'].append(validation_satisfied_by_any_strategy)

			dataset['max_search_time'].append(info_dict['constraint_set_list']['search_time'])

			run_count += 1
		except FileNotFoundError:
			pass






#print(dataset)

print(np.sum(np.array(dataset['best_strategy']) == 0) / float(len(dataset['best_strategy'])))
print(dataset['best_strategy'])
print(len(dataset['best_strategy']))

'''
success_ids = []
for run in range(len(dataset['best_strategy'])):
	if dataset['best_strategy'][run] > 0:
		success_ids.append(run)
		if len(success_ids) == 1000:
			break
'''
success_ids = np.array(list(range(len(dataset['best_strategy']))))

assert len(dataset['success_value']) == len(dataset['best_strategy'])

strategy_recall_test = []
strategy_recall_validation = []

for s in range(1, len(mappnames) + 1):
	current_recall = []
	current_recall_validation = []
	for run in success_ids:
		current_recall.append(dataset['success_value'][run][s])
		current_recall_validation.append(dataset['success_value_validation'][run][s])
	strategy_recall_test.append(current_recall)
	strategy_recall_validation.append(current_recall_validation)

strategy_recall_test = np.array(strategy_recall_test, dtype=np.float)
strategy_recall_validation = np.array(strategy_recall_validation, dtype=np.float)


strategy_time = {}
for s in range(1, len(mappnames) + 1):
	strategy_time[s] = []
	for run in success_ids:
		if not dataset['success_value'][run][s] == True:
			strategy_time[s].append(dataset['times_value'][run][s])


print(strategy_time)


np.sum(np.array(dataset['best_strategy']) == 1)




##get min_max values

min_average_search_time = np.inf
max_fastest = -1
max_coverage_validation = -1
max_finished_test = -1
min_distance_validation = np.inf
min_distance_test = np.inf


def str_float(number):
	return float("{:.2f}".format(number))


for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
	recall = np.sum(finished_test[s + 1]) / float(len(finished_test[s + 1]))

	if max_finished_test < str_float(recall):
		max_finished_test = str_float(recall)

	dist_val = str_float(np.mean(strategy_distance_validation[s + 1])) + str_float(np.std(strategy_distance_validation[s + 1]))
	if min_distance_validation > dist_val:
		min_distance_validation = dist_val

	dist_test = str_float(np.mean(strategy_distance_test[s + 1])) + str_float(np.std(strategy_distance_test[s + 1]))
	if min_distance_test > dist_test:
		min_distance_test = dist_test


latex_string = ''
#for s in range(len(mappnames)):
for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:

	latex_string += str(mappnames[s + 1]) + " & "

	recall = np.sum(finished_test[s + 1]) / float(len(finished_test[s + 1]))

	if max_finished_test == str_float(recall):
		latex_string += "$\\textbf{" + "{:.2f}".format(recall) + '}$'
	else:
		latex_string += "$" + "{:.2f}".format(recall) + '$'


	dist_val = str_float(np.mean(strategy_distance_validation[s + 1])) + str_float(np.std(strategy_distance_validation[s + 1]))
	if min_distance_validation == dist_val:
		latex_string += " & $\\textbf{" + "{:.2f}".format(
			np.mean(strategy_distance_validation[s + 1])) + "} \pm \\textbf{" + "{:.2f}".format(
			np.std(strategy_distance_validation[s + 1])) + '}'
	else:
		latex_string += " & $" + "{:.2f}".format(np.mean(strategy_distance_validation[s + 1])) + " \pm " + "{:.2f}".format(np.std(strategy_distance_validation[s + 1]))

	dist_test = str_float(np.mean(strategy_distance_test[s + 1])) + str_float(np.std(strategy_distance_test[s + 1]))
	if min_distance_test == dist_test:
		latex_string += "$ & $\\textbf{" + "{:.2f}".format(np.mean(strategy_distance_test[s + 1])) + "} \pm \\textbf{" + "{:.2f}".format(
			np.std(strategy_distance_test[s + 1])) + '}'
	else:
		latex_string += "$ & $" + "{:.2f}".format(np.mean(strategy_distance_test[s + 1])) + " \pm " + "{:.2f}".format(np.std(strategy_distance_test[s + 1]))

	latex_string +='$ \\\\ \n'

print("\n\n")

print(latex_string)
