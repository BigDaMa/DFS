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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import backend as K

from sklearn.preprocessing import OneHotEncoder


mappnames = {1:'TPE(Variance)',
			 2: 'TPE($\chi^2$)',
			 3:'TPE(FCBF)',
			 4: 'TPE(Fisher Score)',
			 5: 'TPE(MIM)',
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

experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")

print(experiment_folders)


dataset = {}
dataset['best_strategy'] = []
dataset['features'] = []
dataset['dataset_id'] = []
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

				run_strategies_success_validation[s] = is_successfull_validation(exp_results)
				if run_strategies_success_validation[s]:
					validation_satisfied_by_any_strategy = True

			dataset['success_value'].append(run_strategies_success_test)
			dataset['success_value_validation'].append(run_strategies_success_validation)
			dataset['best_strategy'].append(best_strategy)
			dataset['times_value'].append(run_strategies_times)
			dataset['validation_satisfied'].append(validation_satisfied_by_any_strategy)

			dataset['max_search_time'].append(info_dict['constraint_set_list']['search_time'])
			dataset['features'].append(info_dict['features'])
			dataset['dataset_id'].append(info_dict['dataset_id'])

			run_count += 1
		except FileNotFoundError:
			pass









#todo: balance by class

#print(X_train)
X_train = dataset['features']
X_data = np.array(X_train)
groups = np.array(dataset['dataset_id'])

strategy_success = np.zeros((X_data.shape[0], len(mappnames)))
for c_i in range(len(mappnames)):
	for run in range(X_data.shape[0]):
		strategy_success[run, c_i] = dataset['success_value'][run][c_i+1]

outer_cv = list(GroupKFold(n_splits=20).split(X_data, None, groups=groups))


def get_success_for_fold_predictions(predictions, test_ids):
	all_success = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = test_ids[p_i]
		all_success.append(dataset['success_value'][current_id][current_strategy])
	return all_success


all_runtimes_in_cv_folds = []

real_success_ids = np.where(np.array(dataset['best_strategy']) > 0)[0]

for train_ids, test_ids in outer_cv:

	new_train_ids = []
	for tid in train_ids:
		if tid in real_success_ids:
			new_train_ids.append(tid)
	train_ids = new_train_ids

	def customLoss(y_true, y_pred, **kwargs):
		y = tf.sign(tf.reduce_max(y_pred, axis=-1, keepdims=True) - y_pred)
		y = (y - 1) * (-1)

		return K.sum(y * y_true) * (-1)


	def customLoss2(y_true, y_pred, **kwargs):
		casted_y_true = tf.keras.backend.cast(y_true, dtype='float32')
		return K.sum(tf.math.pow(y_pred, 3) * casted_y_true) * -1

	print(X_data.shape[1])

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(X_data.shape[1], activation='relu', input_dim=X_data.shape[1]))
	model.add(tf.keras.layers.Dense(100, activation='relu'))
	model.add(tf.keras.layers.Dense(len(mappnames), activation='softmax'))
	model.compile(optimizer='adam',
				  loss=customLoss2,
				  metrics=['accuracy'])


	scaler = MinMaxScaler()
	X_train_scaled = scaler.fit_transform(X_data[train_ids])
	X_test_scaled = scaler.transform(X_data[test_ids])

	model.fit(X_train_scaled, strategy_success[train_ids,:], epochs=10, batch_size=1)
	predictions = model.predict_classes(X_test_scaled)
	predictions += 1
	print(predictions)

	succcess_test_fold1 = get_success_for_fold_predictions(predictions, test_ids)
	print("test coverage: " + str(np.sum(succcess_test_fold1) / float(len(succcess_test_fold1))))

print('metalearning cv' + " avg runtime: " + str(np.nanmean(all_runtimes_in_cv_folds)) + " median runtime: " + str(np.nanmedian(all_runtimes_in_cv_folds)) + ' std runtime: ' + str(np.nanstd(all_runtimes_in_cv_folds)))
