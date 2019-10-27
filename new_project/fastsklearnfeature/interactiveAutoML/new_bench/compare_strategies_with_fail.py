from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_sequential_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_forward_seq_search

import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_oneway
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
import time

from fastsklearnfeature.interactiveAutoML.feature_selection.L1Selection import L1Selection
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.RedundancyRemoval import RedundancyRemoval
from fastsklearnfeature.interactiveAutoML.feature_selection.MajoritySelection import MajoritySelection
from fastsklearnfeature.interactiveAutoML.feature_selection.ALSelection import ALSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.HyperOptSelection import HyperOptSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection

from fastsklearnfeature.feature_selection.ComplexityDrivenFeatureConstruction import ComplexityDrivenFeatureConstruction
from fastsklearnfeature.reader.ScikitReader import ScikitReader
from fastsklearnfeature.transformations.MinusTransformation import MinusTransformation
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import multiprocessing as mp
import itertools
from sklearn.ensemble import RandomForestRegressor
import scipy.special
import seaborn as sns
import matplotlib.pyplot as plt
from fastsklearnfeature.configuration.Config import Config


X_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.data', delimiter=' ', header=None).values[:,0:500] [0:100,:]
y_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.labels', delimiter=' ', header=None).values [0:100]


# generate grid
complexity_grid = np.arange(1, X_train.shape[1]+1)
max_acc = 1.0
accuracy_grid = np.arange(0.0, max_acc, max_acc / len(complexity_grid))

def get_estimated_runtimes(old_model = "/tmp/model11_hyperopt.p"):

	grid = list(itertools.product(complexity_grid, accuracy_grid))
	meta_X_data = np.matrix(grid)

	al_model = pickle.load(open(old_model, "rb"))
	runtime_predictions = al_model.predict(meta_X_data)

	df = pd.DataFrame.from_dict(np.array([meta_X_data[:, 0].A1, meta_X_data[:, 1].A1, runtime_predictions]).T)
	df.columns = ['Max Complexity', 'Min Accuracy', 'Estimated Runtime']
	pivotted = df.pivot('Max Complexity', 'Min Accuracy', 'Estimated Runtime')

	return pivotted

#hyperopt_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/model48_run_hyperopt_search.p')
#hyperopt_failure = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/success_model48_run_hyperopt_search.p')

hyperopt_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/new_hyper/model48_run_hyperopt_search.p')
hyperopt_failure = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/new_hyper/success_model48_run_hyperopt_search.p')


alk_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/model46_run_al_k_search.p')
alk_failure = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/success_model46_run_al_k_search.p')

forward_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/model44_run_forward_seq_search.p')
forward_failure = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/success_model44_run_forward_seq_search.p')

rfe_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/model36_run_sequential_search.p')
rfe_failure = get_estimated_runtimes('/home/felix/phd/bench_feature_select/new/success_model36_run_sequential_search.p')



min_matrix = np.array([hyperopt_times.values,
						forward_times.values,
						alk_times.values,
					    rfe_times.values])

min_matrix = np.min(min_matrix, axis=0)

def get_best_scatter(times, successes):
	best_x = []
	best_y = []
	for x in range(min_matrix.shape[0]):
		for y in range(min_matrix.shape[1]):
			if times.values[x, y] <= min_matrix[x, y] and successes.values[x, y]:
				best_x.append(complexity_grid[x])
				best_y.append(accuracy_grid[y])
	return best_x, best_y


forward_best_x, forward_best_y = get_best_scatter(forward_times, forward_failure)
alk_best_x, alk_best_y = get_best_scatter(alk_times, alk_failure)
hyperopt_best_x, hyperopt_best_y = get_best_scatter(hyperopt_times, hyperopt_failure)
rfe_best_x, rfe_best_y = get_best_scatter(rfe_times, rfe_failure)

plt.scatter(forward_best_x, forward_best_y, color='blue', marker='+', label='forward selection')
plt.scatter(alk_best_x, alk_best_y, facecolors='none', edgecolors='r', label='AL k selection')
plt.scatter(hyperopt_best_x, hyperopt_best_y, color='yellow', label='Hyperopt best k selection')
plt.scatter(rfe_best_x, rfe_best_y, color='green', label='Recursive Feature Elimination')
plt.xlabel('complexity')
plt.ylabel('accuracy')
plt.ylim([0.0, 1.0])
plt.legend()
plt.show()



plt.subplot(321)
sns.heatmap(hyperopt_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Hyperopt BestK')

plt.subplot(322)
sns.heatmap(alk_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Active Learning K')

plt.subplot(323)
sns.heatmap(forward_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Forward Selection')

plt.subplot(324)
sns.heatmap(rfe_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Recursive Feature Elimination')

plt.subplots_adjust(hspace=1.5)
plt.show()


plt.subplot(321)
sns.heatmap(hyperopt_failure, cmap='RdBu')
plt.title('Hyperopt BestK')

plt.subplot(322)
sns.heatmap(alk_failure, cmap='RdBu')
plt.title('Active Learning K')

plt.subplot(323)
sns.heatmap(forward_failure, cmap='RdBu')
plt.title('Forward Selection')

plt.subplot(324)
sns.heatmap(rfe_failure, cmap='RdBu')
plt.title('Recursive Feature Elimination')

plt.subplots_adjust(hspace=1.5)
plt.show()

'''
sns.heatmap(hyperopt_times, cmap='RdBu')
plt.title('Hyperopt BestK Time')
plt.show()

sns.heatmap(hyperopt_failure, cmap='RdBu')
plt.title('Hyperopt BestK Success')
plt.show()
'''


'''
forward_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/model29_forward.p')
backward_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/model32_backward.p')
al_k_times = get_estimated_runtimes('/home/felix/phd/bench_feature_select/model29_al_k.p')

min_matrix = np.array([hyperopt_times.values,
						forward_times.values,
						backward_times.values,
						al_k_times.values])

min_matrix = np.min(min_matrix, axis=0)


print(hyperopt_times.values > min_matrix)

plt.subplot(321)
sns.heatmap(hyperopt_times.values > min_matrix, cmap='RdBu')
plt.title('Hyperopt BestK')

plt.subplot(322)
sns.heatmap(forward_times.values > min_matrix, cmap='RdBu')
plt.title('Forward Selection')

plt.subplot(323)
sns.heatmap(backward_times.values > min_matrix, cmap='RdBu')
plt.title('Recursive Feature Elimination')

plt.subplot(324)
sns.heatmap(al_k_times.values > min_matrix, cmap='RdBu')
plt.title('Active Learning for k')

plt.show()

plt.subplot(321)
sns.heatmap(hyperopt_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Hyperopt BestK')

plt.subplot(322)
sns.heatmap(forward_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Forward Selection')

plt.subplot(323)
sns.heatmap(backward_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Recursive Feature Elimination')

plt.subplot(324)
sns.heatmap(al_k_times, cmap='RdBu', vmin=0, vmax=1200)
plt.title('Active Learning for k')

plt.subplots_adjust(hspace=1.5)
plt.show()
'''