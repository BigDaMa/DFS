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
			 12: 'Forward Selection(no ranking)',
			 13: 'Backward Selection(no ranking)',
			 14: 'Forward Floating Selection(no ranking)',
			 15: 'Backward Floating Selection(no ranking)',
			 16: 'RFE(Logistic Regression)'
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


#logs_adult = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_adult.pickle', 'rb'))
#logs_heart = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_heart.pickle', 'rb'))



#get all files from folder


map_observations2file= {}
map_observations2file[100]     = "/home/felix/phd/meta_learn/scaling_experiment/metalearning_data1584708646.3324661scaling_factor_100.pickle"
map_observations2file[1000]    = "/home/felix/phd/meta_learn/scaling_experiment/metalearning_data1584708719.3181086scaling_factor_1000.pickle"
map_observations2file[10000]   = "/home/felix/phd/meta_learn/scaling_experiment/metalearning_data1584708757.6649964scaling_factor_10000.pickle"
map_observations2file[100000]  = "/home/felix/phd/meta_learn/scaling_experiment/metalearning_data1584708801.784663scaling_factor_100000.pickle"
map_observations2file[1000000] = "/home/felix/phd/meta_learn/scaling_experiment/metalearning_data1584708917.893358scaling_factor_1000000.pickle"

#check minimum runs
minimum_runs = 100
for number_observations in [100, 1000, 10000, 100000, 1000000]:
	all_files = glob.glob(map_observations2file[number_observations]) #1hour

	dataset = {}
	for afile in all_files:
		data = pickle.load(open(afile, 'rb'))
		for key in data.keys():
			if not key in dataset:
				dataset[key] = []
			dataset[key].extend(data[key])

	print(str(number_observations) + ": " + str(len(dataset['success_value'])))
	if len(dataset['success_value']) < minimum_runs:
		minimum_runs = len(dataset['success_value'])


print("minimum runs: " + str(minimum_runs))

map_rows2strategy_mean_time = {}

for number_observations in [100, 1000, 10000, 100000, 1000000]:
	all_files = glob.glob(map_observations2file[number_observations]) #1hour

	dataset = {}
	for afile in all_files:
		data = pickle.load(open(afile, 'rb'))
		for key in data.keys():
			if not key in dataset:
				dataset[key] = []
			dataset[key].extend(data[key])

	print(dataset.keys())

	strategy_recall = []
	for s in range(1, len(mappnames) + 1):
		current_recall = []
		for run in range(minimum_runs):
			if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) > 0:
				current_recall.append(dataset['success_value'][run][s][0])
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
		for run in range(minimum_runs):
			if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) > 0 and dataset['success_value'][run][s][0] == True:
				current_time.append(dataset['times_value'][run][s][0])
			else:
				current_time.append(3 * 60 * 60)
		strategy_time.append(current_time)

	strategy_time = np.array(strategy_time, dtype=np.float)

	mean_time = np.mean(strategy_time, axis=1)
	std_time = np.std(strategy_time, axis=1)

	map_rows2strategy_mean_time[number_observations] = mean_time

	print("shape: " + str(strategy_time.shape))

	print(mean_time.shape)


	for s in range(len(mappnames)):
		print(str(mappnames[s+1]) + ": " + str(mean_time[s]) + " std: " + str(std_time[s]))

	##report fastest strategy
	fastest_strategies = np.zeros(len(mappnames))
	for run in range(minimum_runs):
		best_time = np.inf
		best_strategy = -1
		for s in range(1, len(mappnames) + 1):
			if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) > 0 and dataset['success_value'][run][s][0] == True:
				cu_time = dataset['times_value'][run][s][0]
				if cu_time < best_time:
					best_time = cu_time
					best_strategy = s
		fastest_strategies[best_strategy-1] += 1

	fastest_strategies_fraction = fastest_strategies / np.sum(fastest_strategies)

	print("number of rounds: " + str(np.sum(fastest_strategies)))

	print("\n\nfastest: ")
	for s in range(len(mappnames)):
		print(str(mappnames[s+1]) + ": " + str(fastest_strategies_fraction[s]))
	print("\n\n")









	latex_string = ''
	#for s in range(len(mappnames)):
	for s in np.array([11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
		recall = np.sum(strategy_recall[s]) / float(strategy_recall.shape[1])
		latex_string += str(mappnames[s+1]) + " & $" + "{:.0f}".format(mean_time[s]) + " \pm " + "{:.0f}".format(std_time[s]) + "$ && $" + "{:.2f}".format(fastest_strategies_fraction[s]) + "$ && $" + "{:.2f}".format(recall) + '$ \\\\ \n'

	print("\n\n")


	all_runtimes = []
	for run in range(minimum_runs):
		best_runtime = 3 * 60 * 60
		for s in range(1, len(mappnames) + 1):
			if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) > 0 and \
					dataset['success_value'][run][s][0] == True:
				runtime = min(dataset['times_value'][run][s])
				if runtime < best_runtime:
					best_runtime = runtime
		all_runtimes.append(best_runtime)

	latex_string += str('Oracle') + " & $" + "{:.0f}".format(np.mean(all_runtimes)) + " \pm " + "{:.0f}".format(np.std(all_runtimes)) + "$ && $" + "{:.2f}".format(1.0) + "$ && $" + "{:.2f}".format(1.0) + '$ \\\\ \n'

	print(latex_string)

'''
\\addplot+[mark=triangle*] coordinates{(100, 0.12404487133026124) (500, 0.19385924339294433) (1000, 0.3271944999694824) (10000, 3.740761733055115) };
\\addlegendentry{TPE(Mutual Information))}
'''

my_latex = ""
for s in range(len(mappnames)):
	my_latex += '\\addplot+ coordinates{'
	for number_observations in [100, 1000, 10000, 100000, 1000000]:
		my_latex += '(' + str(number_observations) + ',' + str(map_rows2strategy_mean_time[number_observations][s]) + ') '
	my_latex += "};\n\\addlegendentry{"+ str(mappnames[s+1]) +"}\n\n"

print(my_latex)



