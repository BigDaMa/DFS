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
import matplotlib.pyplot as plt

def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

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

map_dataset2name = {}
map_dataset2name['31'] = 'German Credit'
map_dataset2name['802'] = 'Primary Biliary Cirrhosis'
map_dataset2name['1590'] = 'Adult'
map_dataset2name['1461'] = 'Bank Marketing'
map_dataset2name['42193'] = 'COMPAS'
map_dataset2name['1480'] = 'Indian Liver Patient'
#map_dataset2name['804'] = 'hutsof99_logis'
map_dataset2name['42178'] = 'Telco Customer Churn'
map_dataset2name['981'] = 'KDD Internet Usage'
map_dataset2name['40536'] = 'Speed Dating'
map_dataset2name['40945'] = 'Titanic'
map_dataset2name['451'] = 'Irish Educational Transitions'
#map_dataset2name['945'] = 'Kidney'
map_dataset2name['446'] = 'Leptograpsus crabs'
map_dataset2name['1017'] = 'Arrhythmia'
map_dataset2name['957'] = 'Brazil Tourism'
map_dataset2name['41430'] = 'Diabetic Mellitus'
map_dataset2name['1240'] = 'AirlinesCodrnaAdult'
map_dataset2name['1018'] = 'IPUMS Census'
#map_dataset2name['55'] = 'Hepatitis'
map_dataset2name['38'] = 'Thyroid Disease'
map_dataset2name['1003'] = 'Primary Tumor'
map_dataset2name['934'] ='Social Mobility'


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
			 11: 'ES(NR)',
			 12: 'SFS(NR)',
			 13: 'SBS(NR)',
			 14: 'SFFS(NR)',
			 15: 'SBFS(NR)',
			 16: 'RFE(LR)',
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


#logs_adult = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_adult.pickle', 'rb'))
#logs_heart = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_heart.pickle', 'rb'))

experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")

print(experiment_folders)

#get all files from folder

dataset = {}
dataset['best_strategy'] = []
dataset['validation_satisfied'] = []
dataset['dataset_id'] = []


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
			dataset['dataset_id'].append(info_dict['dataset_id'])

			run_count += 1
		except FileNotFoundError:
			pass

joined_strategies = []


map_data_2_coverage = {}
map_data_2_fastest = {}

for run in range(len(dataset['best_strategy'])):

	if dataset['best_strategy'][run] != 0:
		for s in range(1, len(mappnames) + 1):

			if not dataset['dataset_id'][run] in map_data_2_coverage:
				map_data_2_coverage[dataset['dataset_id'][run]] = {}
			if not s in map_data_2_coverage[dataset['dataset_id'][run]]:
				map_data_2_coverage[dataset['dataset_id'][run]][s] = []

			if dataset['success_value'][run][s]==True:
				map_data_2_coverage[dataset['dataset_id'][run]][s].append(True)
			else:
				map_data_2_coverage[dataset['dataset_id'][run]][s].append(False)


			if not dataset['dataset_id'][run] in map_data_2_fastest:
				map_data_2_fastest[dataset['dataset_id'][run]] = {}
			if not s in map_data_2_fastest[dataset['dataset_id'][run]]:
				map_data_2_fastest[dataset['dataset_id'][run]][s] = []

			if s == dataset['best_strategy'][run]:
				map_data_2_fastest[dataset['dataset_id'][run]][s].append(True)
			else:
				map_data_2_fastest[dataset['dataset_id'][run]][s].append(False)

latex = "\\begin{tabular}{@{}l"
for s in range(1, len(mappnames) + 1):
	latex += 'cc'
latex += "@{}}\\toprule\nStrategy "
for s in range(1, len(mappnames) + 1):
	latex += ' & \\multicolumn{2}{c}{' + str(mappnames[s]) + '}'
latex += '\\\\ \n'
for s in range(1, len(mappnames) + 1):
	latex += ' & S & C '
latex += '\\\\ \\midrule \n'

for dataset_id in map_data_2_coverage.keys():
	latex += map_dataset2name[dataset_id]
	for s in range(1, len(mappnames) + 1):
		latex += ' & ' + "{:.2f}".format(np.sum(map_data_2_fastest[dataset_id][s]) / float(len(map_data_2_fastest[dataset_id][s]))) + ' & ' + "{:.2f}".format(np.sum(map_data_2_coverage[dataset_id][s]) / float(len(map_data_2_coverage[dataset_id][s])))
	latex += '\\\\ \n'

print(latex)


my_data = np.zeros((len(map_data_2_coverage), len(mappnames)))

data_keys = list(map_data_2_coverage.keys())
for data_i in range(len(data_keys)):
	dataset_id = data_keys[data_i]
	for s in range(1, len(mappnames) + 1):
		#my_data[data_i, s-1] = np.sum(map_data_2_fastest[dataset_id][s]) / float(len(map_data_2_fastest[dataset_id][s]))
		my_data[data_i, s - 1] = np.sum(map_data_2_coverage[dataset_id][s]) / float(len(map_data_2_coverage[dataset_id][s]))

from sklearn_extra.cluster import KMedoids
kmeans = KMedoids(n_clusters=2, random_state=42, max_iter=100000).fit(my_data)

coverage_representative_datasets = []
for d in range(len(map_data_2_coverage)):
	for c in range(len(kmeans.cluster_centers_)):
		if (my_data[d] == kmeans.cluster_centers_[c]).all():
			print(map_dataset2name[data_keys[d]])
			coverage_representative_datasets.append(data_keys[d])
print(coverage_representative_datasets)

for data_i in range(len(data_keys)):
	dataset_id = data_keys[data_i]
	for s in range(1, len(mappnames) + 1):
		my_data[data_i, s-1] = np.sum(map_data_2_fastest[dataset_id][s]) / float(len(map_data_2_fastest[dataset_id][s]))


from sklearn_extra.cluster import KMedoids
kmeans = KMedoids(n_clusters=2, random_state=42, max_iter=100000).fit(my_data)

fastest_representative_datasets = []
for d in range(len(map_data_2_coverage)):
	for c in range(len(kmeans.cluster_centers_)):
		if (my_data[d] == kmeans.cluster_centers_[c]).all():
			print(map_dataset2name[data_keys[d]])
			fastest_representative_datasets.append(data_keys[d])
print(fastest_representative_datasets)


all_representatives = list(set(coverage_representative_datasets).union(fastest_representative_datasets))

all_representatives = ['446', '42178', '934', '1240']

print(all_representatives)

latex = "\\begin{tabular}{@{}l"
for d in range(len(all_representatives)):
	latex += 'cc'
latex += "@{}}\\toprule\nStrategy "
for d in range(len(all_representatives)):
	latex += ' & \\multicolumn{2}{c}{' + str(map_dataset2name[all_representatives[d]]) + '}'
latex += '\\\\ \n'
for d in range(len(all_representatives)):
	latex += ' & Fastest & Coverage '
latex += '\\\\ \\midrule \n'


fastest = np.zeros(len(all_representatives))
max_coverage = np.zeros(len(all_representatives))

def str_float(number):
	return float("{:.2f}".format(number))

def str_only(number):
	return "{:.2f}".format(number)

for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
	for d in range(len(all_representatives)):
		dataset_id = all_representatives[d]

		current_fastest = np.sum(map_data_2_fastest[dataset_id][s+1]) / float(len(map_data_2_fastest[dataset_id][s+1]))
		current_coverage = np.sum(map_data_2_coverage[dataset_id][s+1]) / float(len(map_data_2_coverage[dataset_id][s+1]))

		if str_float(current_fastest) > fastest[d]:
			fastest[d] = str_float(current_fastest)

		if str_float(current_coverage) > max_coverage[d]:
			max_coverage[d] = str_float(current_coverage)




for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
	latex += mappnames[s+1]
	for d in range(len(all_representatives)):
		dataset_id = all_representatives[d]

		current_fastest = np.sum(map_data_2_fastest[dataset_id][s + 1]) / float(
			len(map_data_2_fastest[dataset_id][s + 1]))
		current_coverage = np.sum(map_data_2_coverage[dataset_id][s + 1]) / float(
			len(map_data_2_coverage[dataset_id][s + 1]))

		if str_float(current_fastest) == fastest[d]:
			latex += ' & \\textbf{' + str_only(current_fastest) + '}'
		else:
			latex += ' & ' + str_only(current_fastest)

		if str_float(current_coverage) == max_coverage[d]:
			latex += ' & \\textbf{' + str_only(current_coverage) + '}'
		else:
			latex += ' & ' + str_only(current_coverage)

	latex += '\\\\ \n'

print(latex)
