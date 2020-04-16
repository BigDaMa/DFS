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
from scipy.stats import pearsonr


mappnames = {1:'var',
			 2: 'chi2',
			 3:'FCBF',
			 4: 'Fisher score',
			 5: 'mutual_info_classif',
			 6: 'MCFS',
			 7: 'ReliefF',
			 8: 'TPE',
             9: 'simulated_annealing',
			 10: 'NSGA-II',
			 11: 'exhaustive',
			 12: 'forward_selection',
			 13: 'backward_selection',
			 14: 'forward_floating_selection',
			 15: 'backward_floating_selection',
			 16: 'recursive_feature_elimination'
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
all_files = glob.glob("/home/felix/phd/meta_learn/new_bugfree/*.pickle")


dataset = {}
for afile in all_files:
	data = pickle.load(open(afile, 'rb'))
	for key in data.keys():
		if not key in dataset:
			dataset[key] = []
		dataset[key].extend(data[key])


print(dataset.keys())


all_fairness = []
all_privacy = []
all_number_features = []
all_number_features_abs = []
all_accuracy = []
all_safety = []

all_runtime = []


for i in range(len(dataset['acc_value'])):
	curr_results = dataset['acc_value']
	if 8 in dataset['acc_value'][i]:
		all_accuracy.append(dataset['acc_value'][i][8][0])
		all_fairness.append(dataset['fair_value'][i][8][0])
		all_safety.append(dataset['robust_value'][i][8][0])
		all_privacy.append(dataset['features'][i][5])
		all_number_features.append(dataset['k_value'][i][8][0])
		all_number_features_abs.append(dataset['k_value'][i][8][0] * dataset['features'][i][14])
		all_runtime.append(dataset['times_value'][i][8][0])


all_privacy = np.array(all_privacy)*-1 #make greater = better

print(len(all_accuracy))

#print(all_number_features_abs)
print(all_privacy)

print('accuracy - safety: ' + str(pearsonr(all_accuracy, all_safety)))
print('accuracy - fairness: ' + str(pearsonr(all_accuracy, all_fairness)))
print('safety - fairness: ' + str(pearsonr(all_safety, all_fairness)))

print('accuracy - privacy: ' + str(pearsonr(all_accuracy, all_privacy)))
print('safety - privacy: ' + str(pearsonr(all_safety, all_privacy)))

print('relation to feature number_rel: ')
print('privacy - number features: ' + str(pearsonr(all_privacy, all_number_features)))
print('safety - number features: ' + str(pearsonr(all_safety, all_number_features)))
print('fairness - number features: ' + str(pearsonr(all_fairness, all_number_features)))
print('accuracy - number features: ' + str(pearsonr(all_accuracy, all_number_features)))


print('relation to search time: ')
print('features - search time: ' + str(pearsonr(all_number_features, all_runtime)))
print('privacy - search time: ' + str(pearsonr(all_privacy, all_runtime)))
print('safety - search time: ' + str(pearsonr(all_safety, all_runtime)))
print('fairness - search time: ' + str(pearsonr(all_fairness, all_runtime)))
print('accuracy - search time: ' + str(pearsonr(all_accuracy, all_runtime)))








