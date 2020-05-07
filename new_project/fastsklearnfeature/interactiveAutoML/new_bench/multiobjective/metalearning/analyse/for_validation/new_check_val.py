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
			 4: 'TPE(Fisher Score)',
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
			 16: 'RFE(Logistic Regression)',
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
dataset['success_value'] = []
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


def is_successfull(exp_results):
	return len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1]['success_test'] == True #also on test satisfied
	#return len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[-1]  # constraints were satisfied on validation set



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


strategy_distance = {}
for s in range(1, len(mappnames) + 1):
	strategy_distance[s] = []

oracle_distance = []
for efolder in experiment_folders:
	run_folders = glob.glob(efolder + "*/")
	for rfolder in run_folders:
		info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))

		distance_of_this_run = []
		for s in range(1, len(mappnames) + 1):
			exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')

			min_loss = np.inf
			best_run = None

			for min_r in range(len(exp_results)):
				if 'loss' in exp_results[min_r] and exp_results[min_r]['loss'] < min_loss:
					min_loss = exp_results[min_r]['loss']
					best_run = min_r

			my_dist = distance_to_constraints_on_test(exp_results, best_run, info_dict)
			strategy_distance[s].append(my_dist)
			distance_of_this_run.append(my_dist)
		oracle_distance.append(min(distance_of_this_run))


run_count = 0
for efolder in experiment_folders:
	run_folders = glob.glob(efolder + "*/")
	for rfolder in run_folders:
		run_strategies_success = {}
		run_strategies_times = {}

		min_time = np.inf
		best_strategy = 0
		for s in range(1, len(mappnames) + 1):
			exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
			if is_successfull(exp_results):
				runtime = exp_results[-1]['final_time']
				if runtime < min_time:
					min_time = runtime
					best_strategy = s

				run_strategies_success[s] = True
				run_strategies_times[s] = runtime
			else:
				run_strategies_success[s] = False
		dataset['success_value'].append(run_strategies_success)
		dataset['best_strategy'].append(best_strategy)
		dataset['times_value'].append(run_strategies_times)

		info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))
		dataset['max_search_time'].append(info_dict['constraint_set_list']['search_time'])

		run_count += 1

print(oracle_distance)


#print(dataset)

print(np.sum(np.array(dataset['best_strategy']) == 0) / float(len(dataset['best_strategy'])))
print(dataset['best_strategy'])

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

strategy_recall = []
for s in range(1, len(mappnames) + 1):
	current_recall = []
	for run in success_ids:
		#if dataset['best_strategy'][run] > 0:
		if dataset['success_value'][run][s] == True:
			current_recall.append(dataset['success_value'][run][s])
		else:
			current_recall.append(False)
	strategy_recall.append(current_recall)

strategy_recall = np.array(strategy_recall, dtype=np.float)
for s in range(len(mappnames)):
	recall = np.sum(strategy_recall[s]) / float(strategy_recall.shape[1])
	print(str(mappnames[s+1]) + ": " + str(recall))

strategy_time = []
for s in range(1, len(mappnames) + 1):
	current_time = []
	for run in success_ids:
		#if dataset['best_strategy'][run] > 0:
		if dataset['success_value'][run][s] == True:
			current_time.append(dataset['times_value'][run][s])
		else:
			current_time.append(dataset['max_search_time'][run])
	strategy_time.append(current_time)

strategy_time = np.array(strategy_time, dtype=np.float)

mean_time = np.mean(strategy_time, axis=1)
std_time = np.std(strategy_time, axis=1)

print("shape: " + str(strategy_time.shape))

print(mean_time.shape)


for s in range(len(mappnames)):
	print(str(mappnames[s+1]) + ": " + str(mean_time[s]) + " std: " + str(std_time[s]))

##report fastest strategy

topk=2
dict_fastest = {}
for topk in [1,2,3]:
	dict_fastest[topk] = np.zeros(len(mappnames))
	for run in success_ids:
		#if dataset['best_strategy'][run] > 0:
		runtimes = np.ones(len(mappnames)) * np.inf
		for s in range(1, len(mappnames) + 1):
			if dataset['success_value'][run][s] == True:
				runtimes[s-1] = dataset['times_value'][run][s]

		topk_strategies = np.argsort(runtimes * -1)[-topk:][::-1]

		dict_fastest[topk][topk_strategies] += 1

	dict_fastest[topk] = dict_fastest[topk] / float(len(success_ids))


for topk in [1,2,3]:
	print("\n\nfastest: " + str(topk))
	for s in range(len(mappnames)):
		print(str(mappnames[s+1]) + ": " + str(dict_fastest[topk][s]))
	print("\n\n")







latex_string = ''
#for s in range(len(mappnames)):
for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
	recall = np.sum(strategy_recall[s]) / float(strategy_recall.shape[1])
	latex_string += str(mappnames[s+1]) + " & $" + "{:.0f}".format(mean_time[s]) + " \pm " + "{:.0f}".format(std_time[s])

	for topk in [1]:
		latex_string += "$ & $" + "{:.2f}".format(dict_fastest[topk][s])
	latex_string += "$ & $" + "{:.3f}".format(recall) + '$'

	latex_string += " & $" + "{:.2f}".format(np.mean(strategy_distance[s+1])) + " \pm " + "{:.2f}".format(np.std(strategy_distance[s+1]))

	latex_string +='$ \\\\ \n'

print("\n\n")



all_runtimes = []
for run in success_ids:
	#if dataset['best_strategy'][run] > 0:
	best_runtime = dataset['max_search_time'][run]
	for s in range(1, len(mappnames) + 1):
		if dataset['success_value'][run][s] == True:
			runtime = dataset['times_value'][run][s]
			if runtime < best_runtime:
				best_runtime = runtime
	all_runtimes.append(best_runtime)


oracle_coverage = np.sum(np.array(dataset['best_strategy']) != 0) / float(len(dataset['best_strategy']))

latex_string += str('Oracle') + " & $" + "{:.0f}".format(np.mean(all_runtimes)) + " \pm " + "{:.0f}".format(np.std(all_runtimes)) + \
				"$ && $" + "{:.2f}".format(oracle_coverage) + "$ && $" + "{:.2f}".format(oracle_coverage) + '$' \
				+ " & $" + "{:.2f}".format(np.mean(oracle_distance)) + " \pm " + "{:.2f}".format(np.std(oracle_distance)) \
				+ '$ \\\\ \n'

print(latex_string)
