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



mappnames = {1:'var', 2: 'chi2', 3:'acc rank', 4: 'robust rank', 5: 'fair rank', 6: 'weighted ranking', 7: 'hyperopt', 8: 'evo'}

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


def print_strategies(results):
	print("all strategies failed: " + str(results[0]) +
		  "\nvar rank: " + str(results[1]) +
		  '\nchi2 rank: ' + str(results[2]) +
		  '\naccuracy rank: ' + str(results[3]) +
		  '\nrobustness rank: ' + str(results[4]) +
		  '\nfairness rank: ' + str(results[5]) +
		  '\nweighted ranking: ' + str(results[6]) +
		  '\nhyperparameter opt: ' + str(results[7]) +
		  '\nevolution: ' + str(results[8])
		  )


#logs_adult = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_adult.pickle', 'rb'))
#logs_heart = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_heart.pickle', 'rb'))



#get all files from folder

# list files "/home/felix/phd/meta_learn/random_configs_eval"
#all_files = glob.glob("/home/felix/phd/meta_learn/random_configs_eval_long/*.pickle")
#all_files = glob.glob("/home/felix/phd/meta_learn/random_configs_with_repair_optimization/*.pickle")
all_files = glob.glob("/home/felix/phd/meta_learn/combine_random_configs/*.pickle") #1hour
#all_files = glob.glob("/home/felix/phd/meta_learn/3h_configs/*.pickle") #1hour


dataset = {}
for afile in all_files:
	data = pickle.load(open(afile, 'rb'))
	for key in data.keys():
		if not key in dataset:
			dataset[key] = []
		dataset[key].extend(data[key])


print(dataset['best_strategy'])
print(len(dataset['best_strategy']))

print(dataset.keys())


#get maximum number of evaluations if a strategy is fastest
eval_strategies = []
for i in range(9):
	eval_strategies.append([])

print(eval_strategies)
for bests in range(len(dataset['best_strategy'])):
	current_best = dataset['best_strategy'][bests]
	if current_best > 0:
		eval_strategies[current_best].append(dataset['evaluation_value'][bests][current_best][0])

print(eval_strategies)

print("max evaluations:")
for i in range(9):
	if len(eval_strategies[i]) > 0:
		print(mappnames[i] + ' min evaluations: ' + str(np.min(eval_strategies[i])) + ' max evaluations: ' + str(np.max(eval_strategies[i])) + ' avg evaluations: ' + str(np.mean(eval_strategies[i])) + ' len evaluations: ' + str(len(eval_strategies[i])))

strategy_recall = []
names = []
strategy_evaluations = []
for s in range(1,9):
	current_recall = []
	eval_number = []
	for run in range(len(dataset['best_strategy'])):
		if dataset['best_strategy'][run] > 0:
			if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) > 0:
				current_recall.append(dataset['success_value'][run][s][0])
				eval_number.append(dataset['evaluation_value'][run][s][0])
			else:
				current_recall.append(False)
				eval_number.append(0)
	strategy_recall.append(current_recall)
	strategy_evaluations.append(eval_number)
	names.append(mappnames[s])


# get number of evaluations when chi2 finds stuff that evo does not
get_eval = []
for i in range(len(strategy_evaluations[0])):
	if strategy_recall[7][i] == False and strategy_recall[1][i] == True:
		get_eval.append(strategy_evaluations[1][i])

print(get_eval)
print(len(get_eval))
print(np.sum(np.array(get_eval) <=100))
print(np.sum(np.array(get_eval) <=100) / len(get_eval))
print(np.mean(np.array(get_eval)))




#print(strategy_recall)
strategy_recall = np.array(strategy_recall, dtype=np.float)


print(np.sum(strategy_recall, axis=1) / float(strategy_recall.shape[1]))


for s in range(8):
	recall = np.sum(strategy_recall[s]) / float(strategy_recall.shape[1])
	print(str(mappnames[s+1]) + ": " + str(recall))
print('\n\n')


for s in range(8):
	recall = np.sum(np.logical_or(strategy_recall[s], strategy_recall[7])) / float(strategy_recall.shape[1])
	print('evo + ' + str(mappnames[s+1]) + ": " + str(recall))
print('\n\n')

for s in range(8):
	recall = np.sum(np.logical_or(np.logical_or(strategy_recall[s], strategy_recall[7]), strategy_recall[1])) / float(strategy_recall.shape[1])
	print('evo + chi2 + ' + str(mappnames[s+1]) + ": " + str(recall))

print('\n\n')
for s in range(8):
	recall = np.sum(np.logical_or(np.logical_or(np.logical_or(strategy_recall[s], strategy_recall[7]), strategy_recall[1]), strategy_recall[0])) / float(strategy_recall.shape[1])
	print('evo + chi2 + var + ' + str(mappnames[s+1]) + ": " + str(recall))

