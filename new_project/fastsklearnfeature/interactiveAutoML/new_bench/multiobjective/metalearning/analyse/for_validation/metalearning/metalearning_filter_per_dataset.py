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
#import openml
from collections.abc import Iterable
import numpy as np
import copy


ranked_by_coverage_decreasing = [3, 14, 12, 2, 4, 11, 9, 5, 10, 8, 1, 7, 6, 16, 15, 13, 17]
ranked_by_coverage_increasing = copy.deepcopy(ranked_by_coverage_decreasing)
ranked_by_coverage_increasing.reverse()


mappnames = {1:'TPE(Variance)',
			 2: 'TPE($\chi^2$)',
			 3:'TPE(FCBF)',
			 4: 'TPE(Fisher)',
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
			 16: 'RFE(LR)',
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


my_random_state = np.random.RandomState(seed=42)

#get all files from folder

#experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")
#experiment_folders = glob.glob("/home/felix/phd2/experiments_restric/*/")
#experiment_folders = glob.glob("/home/felix/phd2/new_experiments_maybe_final/*/")
experiment_folders = glob.glob("/home/felix/phd2/new_experiments/*/")


#experiment_folders = glob.glob("/home/neutatz/data/dfs_experiments/new_experiments_maybe_final/*/")

print(experiment_folders)


dataset = {}
dataset['best_strategy'] = []
dataset['features'] = []
dataset['dataset_id'] = []
dataset['model'] = []
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

mapid2model = {}
mapid2model[0] = 'Logistic Regression'
mapid2model[1] = 'Gaussian Naive Bayes'
mapid2model[2] = 'Decision Tree'

number_ml_scenarios = 10500

run_count = 0
for efolder in experiment_folders:
	run_folders = sorted(glob.glob(efolder + "*/"))
	for rfolder in run_folders:
		try:
			info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))

			if 'model' in info_dict:

				if info_dict['dataset_id'] == '40536' or info_dict['dataset_id'] == '1461':
					continue

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
				dataset['model'].append(mapid2model[info_dict['model']])

				run_count += 1
		except FileNotFoundError:
			pass
		if run_count == number_ml_scenarios:
			break
	if run_count == number_ml_scenarios:
		break


print(np.unique(dataset['dataset_id']))
print(len(np.unique(dataset['dataset_id'])))

datsdets_all = []
for check_datasets in range(len(dataset['success_value'])):
	if np.sum(np.array(list(dataset['success_value'][check_datasets].values()))) > 0:
		datsdets_all.append(dataset['dataset_id'][check_datasets])

print(np.unique(datsdets_all))
print(len(np.unique(datsdets_all)))





#print(logs_regression['features'])



X_train = dataset['features']
y_train = dataset['best_strategy']




real_success_ids = np.where(np.array(y_train) > 0)[0]
success_ids = np.array(list(range(len(y_train))))
print('training size: ' + str(len(real_success_ids)))



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
#outer_cv_all = list(GroupKFold(n_splits=21).split(X_data, None, groups=groups))
outer_cv_all = list(GroupKFold(n_splits=19).split(X_data, None, groups=groups))


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
	for run in range(X_data.shape[0]):
		strategy_success[run, c_i] = dataset['success_value'][run][c_i+1]


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





variance_skew = []
binary_columns_abs = []
binary_columns_rel = []
class_distribution = []

dt_list = []
nb_list = []
lr_list = []

log_rows_list = []
log_cols_list = []

log_search_time = []

for i in range(len(success_ids)):
	current_id = success_ids[i]

	#print(dataset['model'][current_id])
	dt_list.append(dataset['model'][current_id] == 'Decision Tree')
	nb_list.append(dataset['model'][current_id] == 'Gaussian Naive Bayes')
	lr_list.append(dataset['model'][current_id] == 'Logistic Regression')

	log_rows_list.append(np.log(dataset['features'][current_id][13]))
	log_cols_list.append(np.log(dataset['features'][current_id][14]))

	log_search_time.append(np.log(dataset['features'][current_id][6]))

	'''
	if not dataset['dataset_id'][current_id] in save_data:
		X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id = get_fair_data1(dataset['dataset_id'][current_id])
		save_data[dataset['dataset_id'][current_id]] = (X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id)
	else:
		X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id = save_data[dataset['dataset_id'][current_id]]
	print(X_train.shape)
	'''

	'''

	
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
	'''
	


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

X_data = np.hstack([X_data, make_stakeable(dt_list)])
names_features.append('dt_model')

X_data = np.hstack([X_data, make_stakeable(nb_list)])
names_features.append('nb_model')

X_data = np.hstack([X_data, make_stakeable(lr_list)])
names_features.append('lr_model')

'''
X_data = np.hstack([X_data, make_stakeable(log_rows_list)])
names_features.append('log(rows)')

X_data = np.hstack([X_data, make_stakeable(log_cols_list)])
names_features.append('log(cols)')
'''

#X_data = np.hstack([X_data, make_stakeable(log_search_time)])
#names_features.append('log(time)')




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


def get_success_for_fold_predictions_validation(predictions, test_ids):
	all_success = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = success_ids[test_ids[p_i]]
		if dataset['success_value_validation'][current_id][current_strategy] == True:
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
all_success_in_cv_folds_validation = []


all_successful_times = []
all_fastest_strategies = []




strategy_folds_f1 = {} # strategies x datasets
strategy_folds_precision= {} # strategies x datasets
strategy_folds_recall = {} # strategies x datasets

#print('strategy size: ' + str(strategy_folds_f1.shape))


#save models

for my_strategy in range(strategy_success.shape[1]):
	rf_random = RandomForestClassifier(n_estimators=100, class_weight='balanced')
	rf_random.fit(X_data[real_success_ids], strategy_success[real_success_ids][:, my_strategy])

	with open('/tmp/model_strategy' + str(my_strategy) + '.pickle', 'wb+') as f_log:
		pickle.dump(rf_random, f_log, protocol=pickle.HIGHEST_PROTOCOL)

print("\nmodels done :)\n\n")





data2score = {}

store_all_strategies_results = {}
for my_strategy in range(strategy_success.shape[1]):
	store_all_strategies_results[my_strategy] = []

#import openml

my_string_csv = ''

dicttuple2coverage = {}
dataset_id = 0

#calculate weight per dataset
dataset_id2count = {}
for run in range(X_data.shape[0]):
	if np.sum(strategy_success[run, :]) > 0:
		if not groups[run] in dataset_id2count:
			dataset_id2count[groups[run]] = 0
		dataset_id2count[groups[run]] += 1

print(dataset_id2count)


fastest_strategy_across_datasets = {}
fastest_strategy_across_datasets_metalearning = []


relatives_coverages_optimizer = []
relatives_speed_optimizer = []

relative_coverage_strategies= {}
relative_speed_strategies= {}
for my_strategy in range(strategy_success.shape[1]):
	relative_coverage_strategies[my_strategy] = []
	relative_speed_strategies[my_strategy] = []

for train_ids, test_ids in outer_cv_all:


	new_test_ids = []
	for tid in test_ids:
		if tid in real_success_ids:
			new_test_ids.append(tid)
	#test_ids = copy.deepcopy(new_test_ids)


	new_train_ids = []
	for tid in train_ids:
		if tid in real_success_ids:
			new_train_ids.append(tid)
	train_ids = new_train_ids

	#calculate sample_weights

	'''
	sample_weights = []
	for train_iter in range(len(train_ids)):
		dataset_fraction = dataset_id2count[groups[train_ids[train_iter]]] / float(len(train_ids))
		reverse_fraction = 1.0 - dataset_fraction
		sample_weights.append(reverse_fraction)
	'''



	if len(test_ids) > 0:
		test_data_id = np.unique(groups[test_ids])[0]


		##calculate weights per dataset

		my_counts_data_dict = {}
		all_train_N = len(train_ids)
		for sw in range(all_train_N):
			if not groups[train_ids][sw] in my_counts_data_dict:
				my_counts_data_dict[groups[train_ids][sw]] = 0
			my_counts_data_dict[groups[train_ids][sw]] += 1

		sample_weight = []
		for sw in range(all_train_N):
			sample_weight.append(all_train_N - my_counts_data_dict[groups[train_ids][sw]])
		print(sample_weight)



		#my_string_csv += openml.datasets.get_dataset(dataset_id=test_data_id).name + ';'
		my_string_csv += test_data_id + ';'

		predictions_probabilities = np.zeros((len(test_ids), len(mappnames)))

		strategy_sum = np.sum(strategy_success[test_ids, :], axis=1)
		print(str(strategy_sum.shape) + ' ' + str(len(test_ids)))

		print('oracle coverage: ' + str(np.sum(strategy_sum > 0) / float(len(test_ids))))

		#dicttuple2coverage[('oracle', openml.datasets.get_dataset(dataset_id=test_data_id).name)] = np.sum(strategy_sum > 0) / float(len(test_ids))

		dicttuple2coverage[('oracle', test_data_id)] = np.sum(strategy_sum > 0) / float(len(test_ids))

		my_string_csv += str(np.sum(strategy_sum > 0) / float(len(test_ids))) + ';'

		print(np.sum(strategy_success[test_ids, :], axis=1))

		#train one random forest per strategy
		for my_strategy in range(strategy_success.shape[1]):
			if True:#(my_strategy + 1) in choose_among_strategies:


				'''
				transformer = Normalizer().fit(X_data[train_ids][:, my_ids])
				X_scaled = transformer.transform(X_data[train_ids][:, my_ids])
				X_scaled = transformer.transform(X_data[test_ids][:, my_ids])
				sm = SMOTE(random_state=42)
				X_res, y_res = sm.fit_resample(X_scaled, strategy_success[train_ids, my_strategy])
				'''
				X_res = X_data[train_ids][:, my_ids]
				y_res = strategy_success[train_ids, my_strategy]

				rf_random = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=my_random_state)
				#rf_random.fit(X_res, y_res, sample_weight=sample_weights)
				rf_random.fit(X_res, y_res)

				#plot_most_important_features(rf_random, names_features, title=mappnames[my_strategy+1])

				my_x_test = X_data[test_ids][:, my_ids]#X_scaled#X_data[test_ids][:, my_ids]

				print(mappnames[my_strategy+1] + ': ' + str(f1_scorer(rf_random, my_x_test, strategy_success[test_ids, my_strategy])))

				my_predictions = rf_random.predict(my_x_test)

				if len(new_test_ids) > 0:
					new_my_predictions = rf_random.predict(X_data[new_test_ids][:, my_ids])

					if not my_strategy in strategy_folds_f1:
						strategy_folds_f1[my_strategy] = []
						strategy_folds_precision[my_strategy] = []
						strategy_folds_recall[my_strategy] = []

					strategy_folds_f1[my_strategy].append(f1_score(strategy_success[new_test_ids, my_strategy], new_my_predictions))
					strategy_folds_precision[my_strategy].append(precision_score(strategy_success[new_test_ids, my_strategy], new_my_predictions))
					strategy_folds_recall[my_strategy].append(recall_score(strategy_success[new_test_ids, my_strategy], new_my_predictions))

				predictions_probabilities[:, my_strategy] = rf_random.predict_proba(my_x_test)[:, 1]

				print(mappnames[my_strategy + 1] + ' prob : ' + str(np.mean(predictions_probabilities[:, my_strategy])))

				print(mappnames[my_strategy + 1] + ' coverage : ' + str(np.sum(strategy_success[test_ids, my_strategy]) / float(len(test_ids)) ))

				my_string_csv += str(np.sum(strategy_success[test_ids, my_strategy]) / float(len(test_ids))) + ';'
				#dicttuple2coverage[(mappnames[my_strategy + 1], openml.datasets.get_dataset(dataset_id=test_data_id).name)] = np.sum(strategy_success[test_ids, my_strategy]) / float(len(test_ids))
				dicttuple2coverage[
					(mappnames[my_strategy + 1], test_data_id)] = np.sum(
					strategy_success[test_ids, my_strategy]) / float(len(test_ids))

			else:
				predictions_probabilities[:, my_strategy] = np.zeros(len(test_ids))



		#use these

		dataset_id += 1




		predictions = np.zeros(len(predictions_probabilities), dtype=int)
		for row_it in range(len(predictions_probabilities)):
			winner = np.argwhere(predictions_probabilities[row_it,:] == np.max(predictions_probabilities[row_it,:]))
			winner += 1
			if len(winner) > 1:
				print(winner)
				#raise Exception('more than one')

				#rannked_list = [14, 12, 11, 10, 2, 3, 8, 5, 9, 4, 1, 7, 6, 16, 13, 15, 17]
				#rannked_list = [3, 2, 4, 14, 5, 12, 8, 1, 11, 9, 7, 10, 16, 6, 15, 13, 17]
				rannked_list =  ranked_by_coverage_decreasing

				rank_i = 0
				while isinstance(winner, Iterable):
					if rannked_list[rank_i] in winner:
						winner = rannked_list[rank_i]
						break
					rank_i += 1

			predictions[row_it] = winner


		#predictions = np.argmax(predictions_probabilities, axis=1)
		#predictions += 1
		print('prediction_shape: ' + str(predictions.shape))
		print(predictions)
		print(np.array(dataset['best_strategy'])[success_ids[test_ids]])

		runtimes_test_fold = get_runtime_for_fold_predictions(predictions, test_ids)
		#print('mean time:  ' + str(np.mean(runtimes_test_fold)) + ' std: ' + str(np.std(runtimes_test_fold)))

		succcess_test_fold1 = get_success_for_fold_predictions(predictions, test_ids)
		print(succcess_test_fold1)




		rel_coverage = np.sum(succcess_test_fold1) / float(np.count_nonzero(np.array(dataset['best_strategy'])[success_ids[test_ids]]))
		if not np.isnan(rel_coverage):
			relatives_coverages_optimizer.append(rel_coverage)
		else:
			relatives_coverages_optimizer.append(0.0)

		for my_strategy in range(strategy_success.shape[1]):
			store_all_strategies_results[my_strategy].extend(strategy_success[test_ids, my_strategy])

			rel_coverage = np.sum(strategy_success[test_ids, my_strategy]) / float(np.count_nonzero(np.array(dataset['best_strategy'])[success_ids[test_ids]]))
			if not np.isnan(rel_coverage):
				relative_coverage_strategies[my_strategy].append(rel_coverage)
				print(relative_coverage_strategies)
			else:
				relative_coverage_strategies[my_strategy].append(0.0)

		relatives_speed_optimizer.append(np.sum(np.array(dataset['best_strategy'])[test_ids] == predictions) / float(np.count_nonzero(np.array(dataset['best_strategy'])[success_ids[test_ids]])))

		for my_strategy in range(strategy_success.shape[1]):
			rel_speed = np.sum(np.array(dataset['best_strategy'])[test_ids] == my_strategy + 1) / float(np.count_nonzero(np.array(dataset['best_strategy'])[success_ids[test_ids]]))
			relative_speed_strategies[my_strategy].append(rel_speed)




		succcess_test_fold1_test = []
		for p_i in range(len(predictions)):
			succcess_test_fold1_test.append(strategy_success[test_ids[p_i], predictions[p_i] - 1])
			assert succcess_test_fold1[p_i] == succcess_test_fold1_test[p_i], 'ojoh'



		print("test coverage: " + str(np.sum(succcess_test_fold1) / float(len(succcess_test_fold1))))

		#my_string_csv += str(len(test_ids)) + ';'

		my_string_csv += str(np.sum(succcess_test_fold1) / float(len(succcess_test_fold1))) + '\n'
		#dicttuple2coverage[('metalearning', openml.datasets.get_dataset(dataset_id=test_data_id).name)] = np.sum(succcess_test_fold1) / float(len(succcess_test_fold1))
		dicttuple2coverage[('metalearning', test_data_id)] = np.sum(
			succcess_test_fold1) / float(len(succcess_test_fold1))

		#data2score[openml.datasets.get_dataset(dataset_id=test_data_id).name] = np.sum(succcess_test_fold1) / float(len(succcess_test_fold1))

		data2score[test_data_id] = np.sum(succcess_test_fold1) / float(
			len(succcess_test_fold1))

		print(data2score)


		success_validation = get_success_for_fold_predictions_validation(predictions, test_ids)





		#all_successful_times.extend(successful_times)
		all_runtimes_in_cv_folds.extend(runtimes_test_fold)
		all_success_in_cv_folds.extend(succcess_test_fold1)
		#all_fastest_strategies.extend(fastest_fold_here)
		all_success_in_cv_folds_validation.extend(success_validation)


def toS(value):
	return "{:.2f}".format(value)



latex_table = ''
for my_strategy in np.array(ranked_by_coverage_increasing) - 1:
	latex_table += str(mappnames[my_strategy + 1]) + " & $" + toS(np.mean(relative_speed_strategies[my_strategy])) + ' \pm ' +  toS(np.std(relative_speed_strategies[my_strategy])) + '$ '
	latex_table += " & $" + toS(np.mean(relative_coverage_strategies[my_strategy])) + ' \pm ' + toS(np.std(relative_coverage_strategies[my_strategy])) + '$ \\\\ \n'

latex_table += "Optimizer" + " & $" + toS(np.mean(relatives_speed_optimizer)) + ' \pm ' +  toS(np.std(relatives_speed_optimizer)) + '$ '
latex_table += " & $" + toS(np.mean(relatives_coverages_optimizer)) + ' \pm ' + toS(np.std(relatives_coverages_optimizer)) + '$ \\\\ \n'

print(latex_table)
print('\n\n')




for my_strategy in np.array(ranked_by_coverage_increasing) - 1:
	print(str(mappnames[my_strategy + 1]) + ' & $' + str(
		"{:.2f}".format(np.mean(strategy_folds_precision[my_strategy]))) + ' \\pm ' + str(
		"{:.2f}".format(np.std(strategy_folds_precision[my_strategy]))) + '$ & $' + str(
		"{:.2f}".format(np.mean(strategy_folds_recall[my_strategy]))) + ' \\pm ' + str(
		"{:.2f}".format(np.std(strategy_folds_recall[my_strategy]))) + '$ & $' + str(
		"{:.2f}".format(np.mean(strategy_folds_f1[my_strategy]))) + ' \\pm ' + str(
		"{:.2f}".format(np.std(strategy_folds_f1[my_strategy]))) + '$ \\\\')


print(my_string_csv)

print(dicttuple2coverage)