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
			 5: 'TPE(MIM)',
			 6: 'TPE(MCFS)',
			 7: 'TPE(ReliefF)',
			 8: 'TPE(NR)',
             9: 'SA(NR)',
			 10: 'NSGA-II(NR)',
			 11: 'Exhaustive Search(NR)',
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
			if len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1]['success_test'] == True:
			#if len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[-1]:  # constraints were satisfied on validation set
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


print(dataset['best_strategy'])

success_ids = []
for run in range(len(dataset['best_strategy'])):
	if dataset['best_strategy'][run] > 0:
		success_ids.append(run)
		if len(success_ids) == 1000:
			break

success_ids = np.array(list(range(len(dataset['best_strategy']))))

assert len(dataset['success_value']) == len(dataset['best_strategy'])

joined_strategies = []


my_latex = ''

for rounds in range(10):
	best_recall = 0
	best_combo = []
	best_name = ""
	for s in range(1, len(mappnames) + 1):
		new_joined_strategies = copy.deepcopy(joined_strategies)
		new_joined_strategies.append(s)
		current_recall = []
		for run in success_ids:
			#if dataset['best_strategy'][run] > 0:
			found = False
			for js in new_joined_strategies:
				if dataset['success_value'][run][js]==True:
					found = True
					break
			current_recall.append(found)

		calc_recall = np.sum(current_recall) / float(len(current_recall))
		my_string = ''
		for js in new_joined_strategies:
			my_string += mappnames[js] + ' + '
		my_string += str(calc_recall)
		print(my_string)

		if best_recall < calc_recall:
			best_recall = calc_recall
			best_combo = new_joined_strategies
			best_name = my_string
	my_latex += str(len(best_combo)) + "& + " + mappnames[best_combo[-1]] + " && " + "{:.3f}".format(best_recall) + "\\\\ \n"
	joined_strategies = best_combo
	print("\n\n")


print(my_latex)

for s in range(1, len(mappnames) + 1):
	with open('/tmp/' + mappnames[s] + '_success.txt', 'w+') as the_file:
		for run in success_ids:
			if dataset['success_value'][run][s] == True:
				the_file.write(str(run) + '\n')
