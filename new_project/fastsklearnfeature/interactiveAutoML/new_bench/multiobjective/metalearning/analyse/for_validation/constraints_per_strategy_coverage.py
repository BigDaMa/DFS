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






map_constraint_values_per_strategy = {}
map_constraint_values_success = {}

for s in range(1, len(mappnames) + 1):
	map_constraint_values_per_strategy[s] = {}

	map_constraint_values_per_strategy[s]['accuracy'] = []
	map_constraint_values_per_strategy[s]['fairness'] = []
	map_constraint_values_per_strategy[s]['k'] = []
	map_constraint_values_per_strategy[s]['safety'] = []
	map_constraint_values_per_strategy[s]['privacy'] = []
	map_constraint_values_per_strategy[s]['time'] = []





#experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")
#experiment_folders = glob.glob("/home/felix/phd2/experiments_restric/*/")
experiment_folders = glob.glob("/home/felix/phd2/new_experiments_maybe_final/*/")

print(experiment_folders)


dataset = {}
dataset['best_strategy'] = []
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

number_ml_scenarios = 1500


map_constraint_values_per_strategy = {}

run_count = 0
for efolder in experiment_folders:
	run_folders = sorted(glob.glob(efolder + "*/"))
	for rfolder in run_folders:
		try:
			info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))

			accuracy = info_dict['constraint_set_list']['accuracy']
			fairness = info_dict['constraint_set_list']['fairness']
			k = info_dict['constraint_set_list']['k']
			safety = info_dict['constraint_set_list']['robustness']
			privacy = info_dict['constraint_set_list']['privacy']
			time = info_dict['constraint_set_list']['search_time']

			for s in range(1, len(mappnames) + 1):
				if not s in map_constraint_values_per_strategy:
					map_constraint_values_per_strategy[s] = {}
					map_constraint_values_per_strategy[s]['Fairness'] = []
					map_constraint_values_per_strategy[s]['k'] = []
					map_constraint_values_per_strategy[s]['safety'] = []
					map_constraint_values_per_strategy[s]['privacy'] = []


				exp_results = []
				try:
					exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
				except:
					pass

				if fairness > 0.0:
					map_constraint_values_per_strategy[s]['Fairness'].append(is_successfull_validation_and_test(exp_results))

				if k < 1.0:
					map_constraint_values_per_strategy[s]['k'].append(is_successfull_validation_and_test(exp_results))

				if safety > 0.0:
					map_constraint_values_per_strategy[s]['safety'].append(is_successfull_validation_and_test(exp_results))

				if type(privacy) != type(None):
					map_constraint_values_per_strategy[s]['privacy'].append(is_successfull_validation_and_test(exp_results))

			run_count += 1
		except FileNotFoundError:
			pass
		if run_count == number_ml_scenarios:
			break
	if run_count == number_ml_scenarios:
		break




all_constraints = ['Fairness', 'k', 'safety', 'privacy',]




#times it is set and satisfied / times it is set in total

latex = "\\begin{tabular}{@{}l"
for c in range(len(all_constraints)):
	latex += 'c'
latex += "@{}}\\toprule\nStrategy "
for c in range(len(all_constraints)):
	latex += ' & ' + str(all_constraints[c])
latex += '\\\\ \\midrule \n'

max_coverage_per_constraint = np.zeros(len(all_constraints))

my_format = "{:.2f}"
for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]):
	for constraint_i in range(len(all_constraints)):
		my_value = float(my_format.format(np.sum(map_constraint_values_per_strategy[s][all_constraints[constraint_i]]) / float(len(map_constraint_values_per_strategy[s][all_constraints[constraint_i]]))))

		if my_value > max_coverage_per_constraint[constraint_i]:
			max_coverage_per_constraint[constraint_i] = my_value



for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]):
	latex += mappnames[s]
	for constraint_i in range(len(all_constraints)):
		my_value = float(my_format.format(
			np.sum(map_constraint_values_per_strategy[s][all_constraints[constraint_i]]) / float(
				len(map_constraint_values_per_strategy[s][all_constraints[constraint_i]]))))

		if my_value == max_coverage_per_constraint[constraint_i]:
			latex += ' & $\\textbf{' + my_format.format(my_value) + '} $'
		else:
			latex += ' & $' + my_format.format(my_value) + ' $'
	latex += '\\\\ \n'
print(latex)