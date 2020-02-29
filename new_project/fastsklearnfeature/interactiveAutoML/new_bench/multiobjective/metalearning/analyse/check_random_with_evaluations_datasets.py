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

import openml



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











#print(logs_regression['features'])

my_score = make_scorer(time_score2, greater_is_better=False, logs=dataset)
my_recall_score = make_scorer(get_recall, logs=dataset)
my_runtime_score = make_scorer(get_avg_runtime, logs=dataset)
my_optimal_runtime_score = make_scorer(get_optimum_avg_runtime, logs=dataset)

X_train = dataset['features']
y_train = dataset['best_strategy']

meta_classifier = RandomForestClassifier(n_estimators=1000)
#meta_classifier = DecisionTreeClassifier(random_state=0, max_depth=3)
meta_classifier = meta_classifier.fit(X_train, np.array(y_train) == 0)
#meta_classifier = meta_classifier.fit(X_train, y_train)





#meta_classifier = DummyClassifier(strategy="uniform")
#meta_classifier = DummyClassifier(strategy="most_frequent")
#meta_classifier = DummyClassifier(strategy="constant", constant=8)

#scores = cross_val_score(meta_classifier, X_train, np.array(y_train) == 0, cv=10, scoring='f1')
#print('did it fail: ' + str(np.mean(scores)))


success_ids = np.where(np.array(y_train) > 0)[0]
print(success_ids)

print(dataset.keys())

my_score = make_scorer(time_score2, greater_is_better=False, logs=dataset)



#todo: balance by class

#print(X_train)
X_data = np.array(X_train)[success_ids]
y_data = pd.DataFrame(y_train).iloc[success_ids]
groups = np.array(dataset['dataset_id'])[success_ids]




print('number datasets: ' + str(len(np.unique(groups))))

for data_id in np.unique(groups):
	print(str(data_id) + ":  " + str(np.count_nonzero(groups == data_id)))
	dataset = openml.datasets.get_dataset(int(data_id))
	print('name: ' + str(dataset.name))






