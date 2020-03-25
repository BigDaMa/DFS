import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_recall
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import time_score2
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_avg_runtime
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_optimum_avg_runtime

from sklearn.metrics import r2_score
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
from sklearn.metrics import f1_score

import numpy as np
import copy
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
import random
from sklearn.impute import SimpleImputer
from arff2pandas import a2p

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_mcfs
from sklearn.model_selection import train_test_split
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score

from sklearn.calibration import CalibratedClassifierCV



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

names_features = ['accuracy',
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

#all_files = glob.glob("/home/felix/phd/meta_learn/fair_data/*.pickle") #1hour
all_files = glob.glob("/home/felix/phd/meta_learn/new_bugfree/*.pickle") #1hour


dataset = {}
for afile in all_files:
	data = pickle.load(open(afile, 'rb'))
	for key in data.keys():
		if not key in dataset:
			dataset[key] = []
		dataset[key].extend(data[key])


print(dataset['best_strategy'])
print(len(dataset['best_strategy']))

#dict_keys(['features', 'best_strategy', 'ranking_scores_info', 'times_value', 'k_value', 'acc_value', 'fair_value', 'robust_value', 'success_value', 'evaluation_value', 'dataset_id', 'sensitive_attribute_id'])
assert len(dataset['best_strategy']) == len(dataset['times_value'])
assert len(dataset['success_value']) == len(dataset['times_value'])


print(dataset.keys())


#get maximum number of evaluations if a strategy is fastest
eval_strategies = []
for i in range(len(mappnames) + 1):
	eval_strategies.append([])


##calculate best strategy:
best_strategies_real = []
for current_id in range(len(dataset['best_strategy'])):
	best_runtime = dataset['features'][current_id][6]
	best_id = 0
	for s in range(1, len(mappnames) + 1):
		if s in dataset['success_value'][current_id] and len(dataset['success_value'][current_id][s]) > 0 and \
				dataset['success_value'][current_id][s][0] == True:

			runtime = min(dataset['times_value'][current_id][s])
			if runtime < best_runtime:
				best_runtime = runtime
				best_id = s
	best_strategies_real.append(best_id)
print('real best:' + str(best_strategies_real))



print(eval_strategies)
for bests in range(len(dataset['best_strategy'])):
	current_best = dataset['best_strategy'][bests]
	if current_best > 0:
		eval_strategies[current_best].append(dataset['evaluation_value'][bests][current_best][0])

print(eval_strategies)

print("max evaluations:")
for i in range(len(mappnames) + 1):
	if len(eval_strategies[i]) > 0:
		print(mappnames[i] + ' min evaluations: ' + str(np.min(eval_strategies[i])) + ' max evaluations: ' + str(np.max(eval_strategies[i])) + ' avg evaluations: ' + str(np.mean(eval_strategies[i])) + ' len evaluations: ' + str(len(eval_strategies[i])))











#print(logs_regression['features'])

my_score = make_scorer(time_score2, greater_is_better=False, logs=dataset, number_strategiesplus1=len(mappnames)+1)
my_recall_score = make_scorer(get_recall, logs=dataset)
my_runtime_score = make_scorer(get_avg_runtime, logs=dataset)
my_optimal_runtime_score = make_scorer(get_optimum_avg_runtime, logs=dataset, number_strategiesplus1=len(mappnames)+1)

X_train = dataset['features']
y_train = dataset['best_strategy']




success_ids = np.where(np.array(y_train) > 0)[0]
print(success_ids)


new_success_ids = []
for s_i in success_ids:
	delete_b = False
	for strategy_i in dataset['evaluation_value'][s_i].keys():
		if dataset['evaluation_value'][s_i][strategy_i][0] == 1:
			delete_b = True
			break
	if not delete_b:
		new_success_ids.append(s_i)

#success_ids = new_success_ids

print("training size: " + str(len(success_ids)))



my_score = make_scorer(time_score2, greater_is_better=False, logs=dataset, number_strategiesplus1=len(mappnames)+1)


#todo: balance by class


#success_ids = list(range(len(dataset['dataset_id'])))

#print(X_train)
X_data = np.matrix(X_train)[success_ids]
y_data = np.array(y_train)[success_ids]
groups = np.array(dataset['dataset_id'])[success_ids]
outer_cv_all = list(GroupKFold(n_splits=4).split(X_data, None, groups=groups))


strategy_search_times = np.zeros((X_data.shape[0], len(mappnames)))

for c_i in range(len(mappnames)):
	current_strategy = c_i + 1
	for i in range(len(success_ids)):
		current_id = success_ids[i]
		if current_strategy in dataset['success_value'][current_id] and dataset['success_value'][current_id][current_strategy][0] == True  and current_strategy in dataset['times_value'][current_id] and len(
				dataset['times_value'][current_id][current_strategy]) >= 1:
			strategy_search_times[i, c_i] = min(dataset['times_value'][current_id][current_strategy])
		else:
			strategy_search_times[i, c_i] = dataset['features'][current_id][6]


strategy_success = np.zeros((X_data.shape[0], len(mappnames)))

for c_i in range(len(mappnames)):
	current_strategy = c_i + 1
	for current_id in range(X_data.shape[0]):
		if current_strategy in dataset['success_value'][current_id] and dataset['success_value'][current_id][current_strategy][0] == True:
			strategy_success[i, c_i] = True

for c_i in range(len(mappnames)):
	current_strategy = c_i + 1
	for i in range(len(success_ids)):
		current_id = success_ids[i]
		if current_strategy in dataset['success_value'][current_id] and dataset['success_value'][current_id][current_strategy][0] == True:
			strategy_success[i, c_i] = True
			#print("hallo: " + str(i) + ": " + str(c_i))


'''
feature_list.append(hps['accuracy'])
feature_list.append(hps['fairness'])
feature_list.append(hps['k'])
feature_list.append(hps['k'] * X_train.shape[1])
feature_list.append(hps['robustness'])
feature_list.append(cv_privacy)
feature_list.append(hps['search_time'])
feature_list.append(cv_acc - hps['accuracy'])
feature_list.append(cv_fair - hps['fairness'])
feature_list.append(cv_k - hps['k'])
feature_list.append((cv_k - hps['k']) * X_train.shape[1])
feature_list.append(cv_robust - hps['robustness'])
feature_list.append(cv_time)
feature_list.append(X_train.shape[0])#number rows
feature_list.append(X_train.shape[1])#number columns
'''
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_fair_data1
from scipy.stats import skew

save_data = {}

save_data = pickle.load(open("/tmp/save_data.p", "rb"))




'''
variance_skew = []
binary_columns_abs = []
binary_columns_rel = []

for i in range(len(success_ids)):
	current_id = success_ids[i]
	if not dataset['dataset_id'][current_id] in save_data:
		X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id = get_fair_data1(dataset['dataset_id'][current_id])
		save_data[dataset['dataset_id'][current_id]] = (X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id)
	else:
		X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id = save_data[dataset['dataset_id'][current_id]]
	print(X_train.shape)

	
	var_weights = np.array(variance(X_train,y_train))
	var_weights /= np.sum(var_weights)
	#print(var_weights)
	variance_skew.append(skew(var_weights))
	
	
	count_binary_col = 0
	for col_i in range(X_train.shape[1]):
		if len(np.unique(X_train[:, col_i])) <= 2:
			count_binary_col += 1
	binary_columns_abs.append(count_binary_col)
	binary_columns_rel.append(count_binary_col / float(X_train.shape[1]))
	


#pickle.dump(save_data, open("/tmp/save_data.p", "wb"))



def make_stakeable(mylist):
	my_array = np.matrix(mylist)
	my_array = my_array.transpose()
	return my_array

#X_data = np.hstack([X_data, make_stakeable(variance_skew)])
#names_features.append('variance_skew')
#X_data = np.hstack([X_data, make_stakeable(binary_columns_abs)])
#names_features.append('binary_columns_abs')
#X_data = np.hstack([X_data, make_stakeable(binary_columns_rel)])
#names_features.append('binary_columns_rel')

'''



#my_ids = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15]
my_ids = list(range(X_data.shape[1]))



#hyperparameter optimization
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
			   'class_weight': ['balanced']
			   }

def get_runtime_for_fold_predictions(predictions, test_ids):
	all_runtimes = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = success_ids[test_ids[p_i]]
		if current_strategy in dataset['success_value'][current_id] and len(
				dataset['success_value'][current_id][current_strategy]) > 0 and \
				dataset['success_value'][current_id][current_strategy][0] == True:
			all_runtimes.append(min(dataset['times_value'][current_id][current_strategy]))
		else:
			all_runtimes.append(dataset['features'][current_id][6])
	return all_runtimes

def get_success_for_fold_predictions(predictions, test_ids):
	all_success = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = success_ids[test_ids[p_i]]
		if current_strategy in dataset['success_value'][current_id] and len(
				dataset['success_value'][current_id][current_strategy]) > 0 and \
				dataset['success_value'][current_id][current_strategy][0] == True:
			all_success.append(True)
		else:
			all_success.append(False)
	return all_success


def get_is_fastest_for_fold_predictions(predictions, test_ids):
	all_success = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = success_ids[test_ids[p_i]]

		best_time = np.inf
		best_strategy = -1
		for s in range(1, len(mappnames) + 1):
			if s in dataset['success_value'][current_id] and len(dataset['success_value'][current_id][s]) > 0 and \
					dataset['success_value'][current_id][s][0] == True:
				cu_time = dataset['times_value'][current_id][s][0]
				if cu_time < best_time:
					best_time = cu_time
					best_strategy = s

		all_success.append(best_strategy == current_strategy)

	return all_success


all_runtimes_in_cv_folds = []
all_success_in_cv_folds = []
all_fastest_in_cv_folds = []
for train_ids, test_ids in outer_cv_all:
	predictions_probabilities = np.zeros((len(test_ids), len(mappnames)))

	f1_scorer = make_scorer(f1_score, greater_is_better=True)
	for my_strategy in range(strategy_success.shape[1]):

		inner_cv = GroupKFold(n_splits=4).split(X_data[train_ids, :], None, groups=groups[train_ids])
		'''
		rf = RandomForestClassifier()
		rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,
									   cv=inner_cv, verbose=2, random_state=42,
									   n_jobs=-1, scoring=f1_scorer)
		'''
		rf_random = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
		rf_random.fit(X_data[train_ids][:, my_ids], strategy_success[train_ids, my_strategy])

		'''
		calibration = CalibratedClassifierCV(base_estimator=rf_random, cv=5 )
		calibration.fit(X_data[train_ids][:, my_ids], strategy_success[train_ids, my_strategy])
		predictions_probabilities[:, my_strategy] = calibration.predict_proba(X_data[test_ids][:, my_ids])[:,1]
		'''

		print(mappnames[my_strategy+1] + ': ' + str(f1_scorer(rf_random, X_data[test_ids][:, my_ids], strategy_success[test_ids,my_strategy])))

		predictions_probabilities[:, my_strategy] = rf_random.predict_proba(X_data[test_ids][:, my_ids])[:,1]

		print(mappnames[my_strategy + 1] + ' prob : ' + str(np.mean(predictions_probabilities[:, my_strategy])))

	predictions = np.argmax(predictions_probabilities, axis=1)
	predictions += 1
	print(predictions.shape)
	print(predictions)

	runtimes_test_fold = get_runtime_for_fold_predictions(predictions, test_ids)
	print('mean time:  ' + str(np.mean(runtimes_test_fold)) + ' std: ' + str(np.std(runtimes_test_fold)))
	all_runtimes_in_cv_folds.extend(runtimes_test_fold)
	all_success_in_cv_folds.extend(get_success_for_fold_predictions(predictions, test_ids))
	all_fastest_in_cv_folds.extend(get_is_fastest_for_fold_predictions(predictions, test_ids))

print('\n final mean time:  ' + str(np.mean(all_runtimes_in_cv_folds)) + ' std: ' + str(np.std(all_runtimes_in_cv_folds)))
print('\n final coverage:  ' + str(np.sum(all_success_in_cv_folds) / float(len(all_success_in_cv_folds))))
print('\n final fastest:  ' + str(np.sum(all_fastest_in_cv_folds) / float(len(all_fastest_in_cv_folds))))


'''
number_models = 0
for a in range(strategy_success.shape[1]):
	a_strategy = a + 1
	for b in range(strategy_success.shape[1]):

		if a == b:
			continue

		b_strategy = b + 1

		a_faster_b = []
		for s_i in range(len(success_ids)):
			current_id = success_ids[s_i]

			if a_strategy in dataset['success_value'][current_id] and \
			dataset['success_value'][current_id][a_strategy][0] == True:
				if b_strategy in dataset['success_value'][current_id] and \
						dataset['success_value'][current_id][b_strategy][0] == True:
					if dataset['times_value'][current_id][a_strategy][0] < dataset['times_value'][current_id][b_strategy][0]:
						a_faster_b.append(True)
					else:
						a_faster_b.append(False)
				else:
					a_faster_b.append(True)
			else:
				a_faster_b.append(False)
		assert len(a_faster_b) == len(success_ids)

		if len(np.unique(a_faster_b)) < 2:
			continue


		rf_random = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
		rf_random.fit(X_data[train_ids][:, my_ids], np.array(a_faster_b)[train_ids])
		number_models +=1

		print('a: ' + mappnames[a_strategy] + ' better than b: ' + mappnames[b_strategy] + " => " + str(
			f1_scorer(rf_random, X_data[test_ids][:, my_ids], np.array(a_faster_b)[test_ids])))


print("number models: " + str(number_models))
'''

'''
for my_strategy in range(strategy_success.shape[1]):

	print('train ids: ' + str(len(train_ids)))
	train_ids = list(set(train_ids).intersection(np.where(strategy_success[:,my_strategy])[0]))
	print('train ids: ' + str(len(train_ids)))

	test_ids = list(set(test_ids).intersection(np.where(strategy_success[:,my_strategy])[0]))

	new_X_train = X_data[train_ids][:, my_ids]
	new_X_test = X_data[test_ids][:, my_ids]




	meta_classifier = RandomForestRegressor(n_estimators=1000)
	meta_classifier = meta_classifier.fit(new_X_train, strategy_search_times[train_ids, my_strategy])

	for i in range(len(meta_classifier.feature_importances_)):
		print(names_features[i] + ": " + str(meta_classifier.feature_importances_[i]))


	r2_scorer = make_scorer(r2_score, greater_is_better=True, multioutput='uniform_average')
	print(mappnames[my_strategy+1] + ': ' +str(r2_scorer(meta_classifier, new_X_test, strategy_search_times[test_ids, my_strategy])))




meta_classifier = RandomForestClassifier(n_estimators=1000)
meta_classifier = meta_classifier.fit(new_X_train, strategy_success[train_ids])

for i in range(len(meta_classifier.feature_importances_)):
	print(names_features[i] + ": " + str(meta_classifier.feature_importances_[i]))

f1_scorer = make_scorer(f1_score, greater_is_better=True, average='micro')
print(f1_scorer(meta_classifier, new_X_test, strategy_success[test_ids]))
'''


#r2_scorer = make_scorer(r2_score, greater_is_better=True, multioutput='uniform_average')
#print(r2_scorer(meta_classifier, new_X_test, strategy_search_times[test_ids,0]))

