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


mappnames = {1: 'Var',
			 2: 'chi2',
			 3:'FCBF',
			 4: 'Fisher',
			 5: 'MIM',
			 6: 'MCFS',
			 7: 'ReliefF',
			 8: 'TPE',
             9: 'Sim. Anneal.',
			 10: 'NSGA-II',
			 11: 'Exhaustive',
			 12: 'SFS',
			 13: 'SBS',
			 14: 'SFFS',
			 15: 'SBFS',
			 16: 'RFE'
			 }

mappnames_new = {1:'TPE(Variance)',
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
			 12: 'SFS(no ranking)',
			 13: 'SBS(no ranking)',
			 14: 'SFFS(no ranking)',
			 15: 'SBFS(no ranking)',
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

all_files = glob.glob("/home/felix/phd/meta_learn/new_bugfree/*.pickle")


dataset = {}
for afile in all_files:
	data = pickle.load(open(afile, 'rb'))
	for key in data.keys():
		if not key in dataset:
			dataset[key] = []
		dataset[key].extend(data[key])


assert len(dataset['success_value']) == len(dataset['best_strategy'])

joined_strategies = []


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

for run in range(len(dataset['best_strategy'])):

	best_strategy = 0
	fastest_time = np.inf
	for s in range(1, len(mappnames) + 1):
		if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) > 0 and dataset['success_value'][run][s][0] == True:
			if dataset['times_value'][run][s][0] < fastest_time:
				best_strategy = s
				fastest_time = dataset['times_value'][run][s][0]

	accuracy = dataset['features'][run][0]  # accuracy
	fairness = dataset['features'][run][1]  # 'fairness'
	k = dataset['features'][run][3]  # 'k',
	safety = dataset['features'][run][4]  # safety
	privacy = dataset['features'][run][5]  # privacy
	time = dataset['features'][run][6]  # time

	if best_strategy != 0:
		for s in range(1, len(mappnames) + 1):
			if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) > 0 and \
					dataset['success_value'][run][s][0] == True:

				map_constraint_values_per_strategy[s]['accuracy'].append(accuracy)
				map_constraint_values_per_strategy[s]['fairness'].append(fairness)
				map_constraint_values_per_strategy[s]['k'].append(k)
				map_constraint_values_per_strategy[s]['safety'].append(safety)
				map_constraint_values_per_strategy[s]['privacy'].append(privacy)
				map_constraint_values_per_strategy[s]['time'].append(time)

all_constraints = ['accuracy', 'fairness', 'k', 'safety', 'privacy', 'time']
fig, axs = plt.subplots(len(all_constraints))
for constraint_i in range(len(all_constraints)):
	my_list = [map_constraint_values_per_strategy[s][all_constraints[constraint_i]] for s in range(1, len(mappnames) + 1)]
	s_names = [mappnames[s] for s in range(1, len(mappnames) + 1)]
	axs[constraint_i].boxplot(my_list, labels=s_names)
	#axs[constraint_i].hist(my_list[1], bins=100)
	axs[constraint_i].set_title(all_constraints[constraint_i])
plt.subplots_adjust(hspace=1)

plt.show()


latex = "\\begin{tabular}{@{}l"
for c in range(len(all_constraints)):
	latex += 'c'
latex += "@{}}\\toprule\nStrategy "
for c in range(len(all_constraints)):
	latex += ' & ' + str(all_constraints[c])
latex += '\\\\ \\midrule \n'


for s in np.array([11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]):
	latex += mappnames_new[s]
	for constraint_i in range(len(all_constraints)):
		my_format = "{:.2f}"
		if all_constraints[constraint_i] == 'time' or all_constraints[constraint_i] == 'k':
			my_format = "{:.0f}"

		latex += ' & $' + my_format.format(np.mean(map_constraint_values_per_strategy[s][all_constraints[constraint_i]])) + ' \pm ' + my_format.format(np.std(map_constraint_values_per_strategy[s][all_constraints[constraint_i]])) + ' $'
	latex += '\\\\ \n'
print(latex)