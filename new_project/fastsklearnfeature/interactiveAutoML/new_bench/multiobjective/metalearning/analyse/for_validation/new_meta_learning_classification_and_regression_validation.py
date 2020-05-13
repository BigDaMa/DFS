import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold
import glob
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import numpy as np
import copy
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


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









#print(logs_regression['features'])



X_train = dataset['features']
y_train = dataset['best_strategy']




real_success_ids = np.where(np.array(y_train) > 0)[0]
success_ids = np.array(list(range(len(y_train))))
print(success_ids)



new_success_ids = copy.deepcopy(success_ids)

#dont skip
#success_ids = np.arange(len(y_train))

print("training size: " + str(len(success_ids)))





#todo: balance by class


#success_ids = list(range(len(dataset['dataset_id'])))

#print(X_train)
X_data = np.matrix(X_train)[success_ids]
y_data = np.array(y_train)[success_ids]
groups = np.array(dataset['dataset_id'])[success_ids]
outer_cv_all = list(GroupKFold(n_splits=20).split(X_data, None, groups=groups))


strategy_search_times = np.zeros((X_data.shape[0], len(mappnames)))

for c_i in range(len(mappnames)):
	current_strategy = c_i + 1
	for i in range(len(success_ids)):
		current_id = success_ids[i]
		if dataset['success_value'][current_id][current_strategy] == True and current_strategy in dataset['times_value'][current_id]:
			strategy_search_times[i, c_i] = dataset['times_value'][current_id][current_strategy]
		else:
			strategy_search_times[i, c_i] = dataset['features'][current_id][6]


strategy_success = np.zeros((X_data.shape[0], len(mappnames)))

for c_i in range(len(mappnames)):
	current_strategy = c_i + 1
	for current_id in range(X_data.shape[0]):
		if dataset['success_value'][current_id][current_strategy] == True:
			strategy_success[i, c_i] = True

for c_i in range(len(mappnames)):
	current_strategy = c_i + 1
	for i in range(len(success_ids)):
		current_id = success_ids[i]
		if dataset['success_value'][current_id][current_strategy] == True:
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

#save_data = pickle.load(open("/tmp/save_data.p", "rb"))



'''

variance_skew = []
binary_columns_abs = []
binary_columns_rel = []
class_distribution = []

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

	class_distribution.append( min([np.sum(y_train), len(y_train) - np.sum(y_train)]) / float(len(y_train)))
	


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

X_data = np.hstack([X_data, make_stakeable(class_distribution)])
names_features.append('class_distribution')
'''




#my_ids = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15]
my_ids = list(range(X_data.shape[1]))

#7,8,11,12
#to_remove = [7,8,11,12] #constraints only
#to_remove = [8,11,12] #constraints + cv acc
#to_remove = [7,8,11] #constraints + cv time
#to_remove = [7,8,11,12]
to_remove = []

to_remove_names = [names_features[id] for id in to_remove]

for mi in range(len(to_remove)):
	my_ids.remove(to_remove[mi])
	names_features.remove(to_remove_names[mi])

print(names_features)




#hyperparameter optimization
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 5000, num = 10)]
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
		if dataset['success_value'][current_id][current_strategy] == True:
			all_runtimes.append(dataset['times_value'][current_id][current_strategy] + dataset['features'][current_id][12])
		else:
			all_runtimes.append(dataset['features'][current_id][6])
	return all_runtimes

def get_success_for_fold_predictions(predictions, test_ids):
	all_success = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = success_ids[test_ids[p_i]]
		if dataset['success_value'][current_id][current_strategy] == True:
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
			if dataset['success_value'][current_id][s] == True:
				cu_time = dataset['times_value'][current_id][s]
				if cu_time < best_time:
					best_time = cu_time
					best_strategy = s

		all_success.append(best_strategy == current_strategy)

	return all_success


def get_is_topk_fastest_for_fold_predictions(predictions, test_ids, topk):
	all_success = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = success_ids[test_ids[p_i]]

		runtimes = np.ones(len(mappnames)) * np.inf
		for s in range(1, len(mappnames) + 1):
			if dataset['success_value'][current_id][s]== True:
				runtimes[s - 1] = dataset['times_value'][current_id][s]

		topk_strategies = np.argsort(runtimes * -1)[-topk:][::-1]

		all_success.append(current_strategy in topk_strategies)

	return all_success


import operator
def plot_most_important_features(rf_random, names_features, title='importance'):
	importances =  {}
	for name_i in range(len(names_features)):
		importances[names_features[name_i]] = rf_random.feature_importances_[name_i]

	sorted_x = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)

	labels = []
	score = []
	t = 0
	for key, value in sorted_x:
		labels.append(key)
		score.append(value)
		t += 1
		if t == 25:
			break

	ind = np.arange(len(score))
	plt.barh(ind, score, align='center', alpha=0.5)
	plt.yticks(ind, labels)
	plt.title(title)
	plt.show()


f1_scorer = make_scorer(f1_score, greater_is_better=True)


#choose_among_strategies = [1,2,3,8,10,14] #works better
choose_among_strategies = [2,8,14]

all_runtimes_in_cv_folds = []
all_success_in_cv_folds = []
all_fastest_in_cv_folds_dict = {}

for topk in [1, 2, 3]:
	all_fastest_in_cv_folds_dict[topk] = []

strategy_folds_f1 = np.zeros((len(mappnames), len(outer_cv_all))) # strategies x datasets
strategy_folds_precision= np.zeros((len(mappnames), len(outer_cv_all))) # strategies x datasets
strategy_folds_recall = np.zeros((len(mappnames), len(outer_cv_all))) # strategies x datasets

print('strategy size: ' + str(strategy_folds_f1.shape))

dataset_id = 0
for train_ids, test_ids in outer_cv_all:

	'''
	# prune test ids
	nn_test_id = []
	for tt in test_ids:
		if tt in new_success_ids:
			nn_test_id.append(tt)
	test_ids = nn_test_id
	'''

	predictions_probabilities = np.zeros((len(test_ids), len(mappnames)))

	for my_strategy in range(strategy_success.shape[1]):

		if True:#(my_strategy + 1) in choose_among_strategies:

			new_train_ids = []
			for tid in train_ids:
				if tid in real_success_ids:
					new_train_ids.append(tid)
			train_ids = new_train_ids

			rf_random = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
			rf_random.fit(X_data[train_ids][:, my_ids], strategy_success[train_ids, my_strategy])

			#plot_most_important_features(rf_random, names_features, title=mappnames[my_strategy+1])

			my_x_test = X_data[test_ids][:, my_ids]

			print(mappnames[my_strategy+1] + ': ' + str(f1_scorer(rf_random, my_x_test, strategy_success[test_ids, my_strategy])))

			my_predictions = rf_random.predict(my_x_test)
			strategy_folds_f1[my_strategy, dataset_id] = f1_score(strategy_success[test_ids, my_strategy], my_predictions)
			strategy_folds_precision[my_strategy, dataset_id] = precision_score(strategy_success[test_ids, my_strategy], my_predictions)
			strategy_folds_recall[my_strategy, dataset_id] = recall_score(strategy_success[test_ids, my_strategy], my_predictions)

			predictions_probabilities[:, my_strategy] = rf_random.predict_proba(my_x_test)[:, 1]

			print(mappnames[my_strategy + 1] + ' prob : ' + str(np.mean(predictions_probabilities[:, my_strategy])))
		else:
			predictions_probabilities[:, my_strategy] = np.zeros(len(test_ids))

	dataset_id += 1

	predictions = np.argmax(predictions_probabilities, axis=1)
	predictions += 1
	print(predictions.shape)
	print(predictions)
	print(np.array(dataset['best_strategy'])[success_ids[test_ids]])

	runtimes_test_fold = get_runtime_for_fold_predictions(predictions, test_ids)
	print('mean time:  ' + str(np.mean(runtimes_test_fold)) + ' std: ' + str(np.std(runtimes_test_fold)))
	all_runtimes_in_cv_folds.extend(runtimes_test_fold)
	all_success_in_cv_folds.extend(get_success_for_fold_predictions(predictions, test_ids))

	for topk in [1,2,3]:
		all_fastest_in_cv_folds_dict[topk].extend(get_is_topk_fastest_for_fold_predictions(predictions, test_ids, topk))

print('\n final mean time:  ' + str(np.mean(all_runtimes_in_cv_folds)) + ' std: ' + str(np.std(all_runtimes_in_cv_folds)))
print('\n final coverage:  ' + str(np.sum(all_success_in_cv_folds) / float(len(all_success_in_cv_folds))))

for topk in [1,2,3]:
	print('\n final fastest topk=' + str(topk) + ": " + str(np.sum(all_fastest_in_cv_folds_dict[topk]) / float(len(success_ids))))


print("average f1 scores: " + str(np.mean(strategy_folds_f1, axis=1)))
print("average precision scores: " + str(np.mean(strategy_folds_precision, axis=1)))
print("average recall scores: " + str(np.mean(strategy_folds_recall, axis=1)))

for my_strategy in np.array([11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
	print(str(mappnames[my_strategy + 1]) + ' & $' + str(
		"{:.2f}".format(np.mean(strategy_folds_precision, axis=1)[my_strategy])) + ' \\pm ' + str(
		"{:.2f}".format(np.std(strategy_folds_precision, axis=1)[my_strategy])) + '$ & $' + str(
		"{:.2f}".format(np.mean(strategy_folds_recall, axis=1)[my_strategy])) + ' \\pm ' + str(
		"{:.2f}".format(np.std(strategy_folds_recall, axis=1)[my_strategy])) + '$ & $' + str(
		"{:.2f}".format(np.mean(strategy_folds_f1, axis=1)[my_strategy])) + ' \\pm ' + str(
		"{:.2f}".format(np.std(strategy_folds_f1, axis=1)[my_strategy])) + '$ \\\\')

