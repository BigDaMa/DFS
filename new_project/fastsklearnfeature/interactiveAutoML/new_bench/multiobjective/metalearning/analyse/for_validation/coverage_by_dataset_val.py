import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_recall
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import time_score2
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import \
	get_avg_runtime
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import \
	get_optimum_avg_runtime

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
	is_efficient = np.ones(costs.shape[0], dtype=bool)
	for i, c in enumerate(costs):
		if is_efficient[i]:
			is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
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
# map_dataset2name['804'] = 'hutsof99_logis'
map_dataset2name['42178'] = 'Telco Customer Churn'
map_dataset2name['981'] = 'KDD Internet Usage'
map_dataset2name['40536'] = 'Speed Dating'
map_dataset2name['40945'] = 'Titanic'
map_dataset2name['451'] = 'Irish Educational Transitions'
# map_dataset2name['945'] = 'Kidney'
map_dataset2name['446'] = 'Leptograpsus crabs'
map_dataset2name['1017'] = 'Arrhythmia'
map_dataset2name['957'] = 'Brazil Tourism'
map_dataset2name['41430'] = 'Diabetic Mellitus'
map_dataset2name['1240'] = 'AirlinesCodrnaAdult'
map_dataset2name['1018'] = 'IPUMS Census'
# map_dataset2name['55'] = 'Hepatitis'
map_dataset2name['38'] = 'Thyroid Disease'
map_dataset2name['1003'] = 'Primary Tumor'
map_dataset2name['934'] = 'Social Mobility'

mappnames = {1: 'TPE(Variance)',
			 2: 'TPE($\chi^2$))',
			 3: 'TPE(FCBF))',
			 4: 'TPE(Fisher Score))',
			 5: 'TPE(Mutual Information))',
			 6: 'TPE(MCFS))',
			 7: 'TPE(ReliefF))',
			 8: 'TPE(no ranking)',
			 9: 'Simulated Annealing(no ranking)',
			 10: 'NSGA-II(no ranking)',
			 11: 'Exhaustive Search(no ranking)',
			 12: 'Forward Selection(no ranking)',
			 13: 'Backward Selection(no ranking)',
			 14: 'Forward Floating Selection(no ranking)',
			 15: 'Backward Floating Selection(no ranking)',
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


experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")

print(experiment_folders)

dataset = {}
dataset['best_strategy'] = []
dataset['success_value'] = []
dataset['times_value'] = []
dataset['max_search_time'] = []


def load_pickle(fname):
	data = []
	with open(fname, "rb") as f:
		while True:
			try:
				data.append(pickle.load(f))
			except EOFError:
				break
	return data


all_test_scores = []

# get test score for optimal val score
for efolder in experiment_folders:
	run_folders = glob.glob(efolder + "*/")
	for rfolder in run_folders:
		run_strategies_success = {}
		run_strategies_times = {}

		for s in range(1, len(mappnames) + 1):
			exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
			min_loss = np.inf
			best_run = None

			for min_r in range(len(exp_results)):
				if 'loss' in exp_results[min_r] and  exp_results[min_r]['loss'] < min_loss:
					min_loss = exp_results[min_r]['loss']
					best_run = min_r


			if type(best_run) != type(None):
				test_acc = exp_results[best_run]['test_acc']
				test_fair = exp_results[best_run]['test_fair']
				test_simplicity = 1.0 - exp_results[best_run]['cv_number_features']
				test_safety = exp_results[best_run]['test_robust']

				all_test_scores.append((test_acc, test_fair, test_simplicity, test_safety))

print(all_test_scores)

import plotly.graph_objects as go

categories = ['accuracy', 'fairness', 'simplicity', 'safety']

key = 'Adult'

value = all_test_scores

fig = go.Figure()

mask = is_pareto_efficient_simple(np.array(value) * -1)

max_value = np.zeros(len(categories))
max_id = np.zeros(len(categories))
for vi in range(len(value)):
	if mask[vi]:
		for ei in range(len(categories)):
			if max_value[ei] < value[vi][ei]:
				max_value[ei] = value[vi][ei]
				max_id[ei] = vi

for vi in range(len(value)):
	if mask[vi]:
		if vi in max_id:
			fig.add_trace(go.Scatterpolar(
				r=value[vi],
				theta=categories,
				fill='toself',
				name='set' + str(vi)
			))

fig.update_layout(
	title=str(key),
	polar=dict(
		radialaxis=dict(
			visible=True,
			range=[0, 1]
		)),
	showlegend=False
)

fig.write_html('/tmp/radar_chart_' + str(key) + '.html', auto_open=False)