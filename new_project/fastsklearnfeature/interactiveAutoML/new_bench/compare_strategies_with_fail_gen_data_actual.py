from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_sequential_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_info
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import multiprocessing as mp
import itertools
from sklearn.ensemble import RandomForestRegressor
import scipy.special
import seaborn as sns
import matplotlib.pyplot as plt
from fastsklearnfeature.configuration.Config import Config


#X_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.data', delimiter=' ', header=None).values[:,0:500] [0:100,:]
#y_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.labels', delimiter=' ', header=None).values [0:100]

'''
X_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000]
y_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.labels', delimiter=' ', header=None).values
data_name = 'ARCENE'
my_path = "/home/felix/phd/feature_constraints/experiments_arcene/"
onehot = False
'''

'''
data = pd.read_csv(Config.get('data_path') + '/musk/musk.csv', delimiter=',', header=0)
y_train = data['class']
X_train = data[data.columns.difference(['class', 'ID', 'molecule_name', 'conformation_name'])].values
data_name = 'musk'
'''


data = pd.read_csv(Config.get('data_path') + '/breastTumor/breastTumor.csv', delimiter=',', header=0)
y_train = data['binaryClass'].values
X_train = data[data.columns.difference(['binaryClass'])].values
data_name = 'breastTumor'
my_path = "/home/felix/phd/feature_constraints/experiments_actual_tumor/"
onehot=True



'''
data = pd.read_csv(Config.get('data_path') + '/promoters/dataset_106_molecular-biology_promoters.csv', delimiter=',', header=0)
y_train = data['class'].values
X_train = data[data.columns.difference(['class', 'instance'])].values
data_name = 'promoters'
my_path = "/home/felix/phd/feature_constraints/experiments_promoters/"
onehot = True
'''


'''
X_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.data', delimiter=' ', header=None).values[:,0:500]
y_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.labels', delimiter=' ', header=None).values
data_name = 'madelon'
my_path = "/home/felix/phd/feature_constraints/experiments_madelon_no_one/"
onehot = False
'''


if onehot:
	xshape = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(X_train).shape[1]
else:
	xshape = X_train.shape[1]




import glob

file_lists = glob.glob(my_path + "*.p")

print(file_lists)


mapfiles = {}

#get max iteration per selection method

from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for file in file_lists:
	if not 'success_actual_results' in file:
		my_dict = pickle.load(open(file, "rb"))
		print(len(my_dict))
		print(my_dict)

		runtimes = []
		accuracies = []
		complexities = []
		for key, value in my_dict.items():
			if value < 1199:
				accuracies.append(key[0])
				complexities.append(key[1])
				runtimes.append(value)


		ax.scatter(accuracies, complexities, runtimes)

ax.set_xlabel('Accuracy')
ax.set_ylabel('Complexity')
ax.set_zlabel('Runtime')
plt.title(file)

plt.show()






'''
for data in mapfiles.keys():

	success_models = []
	runtime_models = []
	strategy_names = []

	all_strategies = list(mapfiles[data].keys())
	all_strategies.sort()
	for strategy in all_strategies:
		runtime_models.append(get_estimated_runtimes(my_path + 'model' + str(mapfiles[data][strategy]) +'_run_'+ str(strategy) +'_data_'+ str(data) +'.p'))
		success_models.append(get_estimated_runtimes(my_path + 'success_model' + str(mapfiles[data][strategy]) +'_run_'+ str(strategy) +'_data_'+ str(data) +'.p'))
		strategy_names.append(strategy)


	min_matrix = np.array([runtime_model.values for runtime_model in runtime_models])
	min_matrix = np.min(min_matrix, axis=0)

'''



'''
# generate grid
complexity_grid = np.arange(1, xshape+1)
max_acc = 1.0
accuracy_grid = np.arange(0.0, max_acc, max_acc / 100.0)

def get_estimated_runtimes(old_model = "/tmp/model11_hyperopt.p"):

	grid = list(itertools.product(complexity_grid, accuracy_grid))
	meta_X_data = np.matrix(grid)

	al_model = pickle.load(open(old_model, "rb"))
	runtime_predictions = al_model.predict(meta_X_data)

	df = pd.DataFrame.from_dict(np.array([meta_X_data[:, 0].A1, meta_X_data[:, 1].A1, runtime_predictions]).T)
	df.columns = ['Max Complexity', 'Min Accuracy', 'Estimated Runtime']
	pivotted = df.pivot('Max Complexity', 'Min Accuracy', 'Estimated Runtime')

	return pivotted



####
# get files
####
import glob

file_lists = glob.glob(my_path + "*.p")

print(file_lists)


mapfiles = {}

#get max iteration per selection method
for file in file_lists:
	if 'success_model' in file:
		tokens = file.split("success_model",1)[1]
		tokens = tokens.split('_run_',1)
		progess_id = int(tokens[0])
		tokens = tokens[1].split('_data_', 1)
		strategy = tokens[0]
		data = tokens[1].split('.p', 1)[0]

		if not data in mapfiles:
			mapfiles[data] = {}
		if not strategy in mapfiles[data]:
			mapfiles[data][strategy] = progess_id

		if mapfiles[data][strategy] < progess_id:
			mapfiles[data][strategy] = progess_id


for data in mapfiles.keys():

	success_models = []
	runtime_models = []
	strategy_names = []

	all_strategies = list(mapfiles[data].keys())
	all_strategies.sort()
	for strategy in all_strategies:
		runtime_models.append(get_estimated_runtimes(my_path + 'model' + str(mapfiles[data][strategy]) +'_run_'+ str(strategy) +'_data_'+ str(data) +'.p'))
		success_models.append(get_estimated_runtimes(my_path + 'success_model' + str(mapfiles[data][strategy]) +'_run_'+ str(strategy) +'_data_'+ str(data) +'.p'))
		strategy_names.append(strategy)


	min_matrix = np.array([runtime_model.values for runtime_model in runtime_models])
	min_matrix = np.min(min_matrix, axis=0)


	for runtime_model in runtime_models:
		sum_runtime = np.sum(runtime_model.values)
		print(strategy + ": difference to minimum in seconds: " + str(sum_runtime - np.sum(min_matrix)))

	def get_best_scatter(times, successes):
		best_x = []
		best_y = []
		for x in range(min_matrix.shape[0]):
			for y in range(min_matrix.shape[1]):
				if times.values[x, y] <= min_matrix[x, y] and successes.values[x, y]:
					best_x.append(complexity_grid[x])
					best_y.append(accuracy_grid[y])
		return best_x, best_y


	for i in range(len(runtime_models)):
		x, y = get_best_scatter(runtime_models[i], success_models[i])
		plt.scatter(x, y, label=strategy_names[i])
	plt.xlabel('complexity')
	plt.ylabel('accuracy')
	plt.ylim([0.0, 1.0])
	plt.legend()
	plt.title('data: ' + str(data))
	plt.show()

	max_runtime = np.max(np.array([runtime_model.values for runtime_model in runtime_models]))
	print("max: " + str(max_runtime))

	for i in range(len(runtime_models)):
		plt.subplot(4,4,1 + i)
		sns.heatmap(runtime_models[i], cmap='RdBu', vmin=0, vmax=max_runtime)
		plt.title(strategy_names[i])

	plt.subplots_adjust(hspace=1.5)
	plt.show()

	for i in range(len(runtime_models)):
		plt.subplot(4,4,1 + i)
		sns.heatmap(success_models[i], cmap='RdBu')
		plt.title(strategy_names[i])

	plt.subplots_adjust(hspace=1.5)
	plt.show()

'''
