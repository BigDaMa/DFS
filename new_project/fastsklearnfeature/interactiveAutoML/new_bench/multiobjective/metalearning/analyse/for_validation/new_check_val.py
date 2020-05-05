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

'''
Exhaustive Search & $x \pm y$ && x\\
Forward Selection & $x \pm y$ && x\\
Backward Selection & $x \pm y$ && x\\
Forward Floating Selection & $x \pm y$ && x\\
Backward Floating Selection & $x \pm y$ && x\\
Recursive Feature Elimination & $x \pm y$ && x\\
Hyperopt(KBest(Fisher Score)) & $x \pm y$ && x\\
Hyperopt(KBest(ReliefF)) & $x \pm y$ && x\\
Hyperopt(KBest(Mutual Information)) & $x \pm y$ && x\\
Hyperopt(KBest(FCBF)) & $x \pm y$ && x\\
Hyperopt(KBest(MCFS)) & $x \pm y$ && x\\
Hyperopt(KBest(Variance)) & $x \pm y$ && x\\
Hyperopt(KBest($\chi^2$)) & $x \pm y$ && x\\
Ranking-free Hyperopt & $x \pm y$ && x\\
Ranking-free Simulated Annealing & $x \pm y$ && x\\
Ranking-free NSGA-II & $x \pm y$ && x\\ \midrule
Meta-learned Strategy Choice & $x \pm y$ && x\\
'''


mappnames = {1:'TPE(Variance)',
			 2: 'TPE($\chi^2$)',
			 3:'TPE(FCBF)',
			 4: 'TPE(Fisher Score)',
			 5: 'TPE(Mutual Information)',
			 6: 'TPE(MCFS)',
			 7: 'TPE(ReliefF)',
			 8: 'TPE(no ranking)',
             9: 'Simulated Annealing(no ranking)',
			 10: 'NSGA-II(no ranking)',
			 11: 'Exhaustive Search(no ranking)',
			 12: 'SFS(no ranking)',
			 13: 'SBS(no ranking)',
			 14: 'SFFS(no ranking)',
			 15: 'SBFS(no ranking)',
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


def load_pickle(fname):
	data = []
	with open(fname, "rb") as f:
		while True:
			try:
				data.append(pickle.load(f))
			except EOFError:
				break
	return data

for efolder in experiment_folders:
	run_folders = glob.glob(efolder + "*/")
	for rfolder in run_folders:
		run_strategies_success = {}
		run_strategies_times = {}

		min_time = np.inf
		best_strategy = 0
		for s in range(1, len(mappnames) + 1):
			exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
			#if len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1]['success_test'] == True:
			if len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[-1]:  # constraints were satisfied on validation set
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


#print(dataset)

print(np.sum(np.array(dataset['best_strategy']) == 0) / float(len(dataset['best_strategy'])))
print(dataset['best_strategy'])

success_ids = []
for run in range(len(dataset['best_strategy'])):
	if dataset['best_strategy'][run] > 0:
		success_ids.append(run)
		if len(success_ids) == 1000:
			break

success_ids = np.array(list(range(len(dataset['best_strategy']))))

assert len(dataset['success_value']) == len(dataset['best_strategy'])

strategy_recall = []
for s in range(1, len(mappnames) + 1):
	current_recall = []
	for run in success_ids:
		if dataset['best_strategy'][run] > 0:
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
		if dataset['best_strategy'][run] > 0:
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
		if dataset['best_strategy'][run] > 0:
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







#TODO: calculate average distance to satisfying the constraint

latex_string = ''
#for s in range(len(mappnames)):
for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
	recall = np.sum(strategy_recall[s]) / float(strategy_recall.shape[1])
	latex_string += str(mappnames[s+1]) + " & $" + "{:.0f}".format(mean_time[s]) + " \pm " + "{:.0f}".format(std_time[s])

	for topk in [1]:
		latex_string += "$ & $" + "{:.2f}".format(dict_fastest[topk][s])
	latex_string += "$ & $" + "{:.3f}".format(recall) + '$ \\\\ \n'

print("\n\n")


all_runtimes = []
for run in success_ids:
	if dataset['best_strategy'][run] > 0:
		best_runtime = dataset['max_search_time'][run]
		for s in range(1, len(mappnames) + 1):
			if dataset['success_value'][run][s] == True:
				runtime = dataset['times_value'][run][s]
				if runtime < best_runtime:
					best_runtime = runtime
		all_runtimes.append(best_runtime)


#TODO: recalculate oracle
#latex_string += str('Oracle') + " & $" + "{:.0f}".format(np.mean(all_runtimes)) + " \pm " + "{:.0f}".format(np.std(all_runtimes)) + "$ && $" + "{:.2f}".format(1.0) + "$ && $" + "{:.2f}".format(1.0) + '$ \\\\ \n'

print(latex_string)
