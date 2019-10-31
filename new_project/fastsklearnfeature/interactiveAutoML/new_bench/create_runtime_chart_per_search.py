from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_sequential_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_forward_seq_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_al_k_search

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
from sklearn.ensemble import RandomForestClassifier
import scipy.special
import seaborn as sns
import matplotlib.pyplot as plt
from fastsklearnfeature.configuration.Config import Config


def run_experiments_for_strategy(X_train, y_train, data_name, my_search_strategy = run_hyperopt_search, max_time = 20 * 60):

	name = my_search_strategy.__name__

	# generate grid
	complexity_grid = np.arange(1, X_train.shape[1]+1)
	max_acc = 1.0
	#accuracy_grid = np.arange(0.0, max_acc, max_acc / len(complexity_grid))
	accuracy_grid = np.arange(0.0, max_acc, max_acc / 100.0)


	#print(complexity_grid)
	#print(accuracy_grid)

	grid = list(itertools.product(complexity_grid, accuracy_grid))

	print(grid)

	#print(len(grid))

	meta_X_data = np.matrix(grid)


	#run 10 random combinations

	random_combinations = 10
	ids = []
	#ids = np.random.choice(len(grid), size=random_combinations, replace=False, p=None)

	for i in range(0, len(accuracy_grid), int(len(accuracy_grid)/float(random_combinations))):
		ids.append(len(grid) - i)



	kfold = StratifiedKFold(n_splits=10, shuffle=False)
	scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

	meta_X_train = np.zeros((random_combinations, 2))
	runtimes = []
	success_check = []


	for rounds in range(20):
		print("number of ids: " + str(len(ids)))
		for i in range(len(ids)):
			complexity = meta_X_data[ids[i], 0]
			accuracy = meta_X_data[ids[i], 1]
			print("min acc: " + str(accuracy))
			try:
				runtime = my_search_strategy(X_train, y_train,
												model=DecisionTreeClassifier(),
												kfold=copy.deepcopy(kfold),
												scoring=scoring,
												max_complexity=int(complexity),
												min_accuracy=accuracy,
												fit_time_out=max_time)
				success_check.append(True)

			except:
				runtime = max_time
				success_check.append(False)
				print("did not find a solution")
			print(runtime)

			if rounds==0:
				meta_X_train[i] = meta_X_data[ids[i]]
			else:
				meta_X_train = np.vstack([meta_X_train, meta_X_data[ids[i]]])
			runtimes.append(runtime)

		al_model = RandomForestRegressor(n_estimators=10)
		al_model.fit(meta_X_train, runtimes)

		pfile = open("/tmp/model" + str(meta_X_train.shape[0]) + "_" + name + '_data_' + data_name +".p", "wb")
		pickle.dump(al_model, pfile)
		pfile.flush()
		pfile.close()

		print(runtimes)

		# calculate uncertainty of predictions for sampled pairs
		predictions = []
		for tree in range(al_model.n_estimators):
			predictions.append(al_model.estimators_[tree].predict(meta_X_data))

		print(predictions)

		uncertainty = np.matrix(np.std(np.matrix(predictions).transpose(), axis=1)).A1

		print('mean uncertainty: ' + str(np.average(uncertainty)))

		uncertainty_sorted_ids = np.argsort(uncertainty * -1)
		ids = [uncertainty_sorted_ids[0]]

		#predict search failure
		if len(np.unique(np.array(success_check))) == 2:
			al_success_model = RandomForestClassifier(n_estimators=10)
			al_success_model.fit(meta_X_train, success_check)

			pfile = open("/tmp/success_model" + str(meta_X_train.shape[0]) + "_" + name + '_data_' + data_name +".p", "wb")
			pickle.dump(al_success_model, pfile)
			pfile.flush()
			pfile.close()

			# calculate uncertainty of predictions for sampled pairs
			predictions = []
			for tree in range(al_success_model.n_estimators):
				predictions.append(al_success_model.estimators_[tree].predict_proba(meta_X_data)[:, 0])

			print(predictions)

			uncertainty = np.matrix(np.std(np.matrix(predictions).transpose(), axis=1)).A1

			print('mean uncertainty: ' + str(np.average(uncertainty)))

			uncertainty_sorted_ids = np.argsort(uncertainty * -1)
			ids.append(uncertainty_sorted_ids[0])




		runtime_predictions = al_model.predict(meta_X_data)

		df = pd.DataFrame.from_dict(np.array([meta_X_data[:, 0].A1, meta_X_data[:, 1].A1, runtime_predictions]).T)
		df.columns = ['Max Complexity', 'Min Accuracy', 'Estimated Runtime']
		pivotted = df.pivot('Max Complexity', 'Min Accuracy', 'Estimated Runtime')
		sns_plot = sns.heatmap(pivotted, cmap='RdBu')
		fig = sns_plot.get_figure()
		fig.savefig("/tmp/output" + str(meta_X_train.shape[0]) + "_" + name + '_data_' + data_name +".png", bbox_inches='tight')
		plt.clf()

'''
X_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000][0:100,:]
y_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.labels', delimiter=' ', header=None).values[0:100]
run_experiments_for_strategy(X_train, y_train, 'arcene_sample', run_hyperopt_search, max_time = 20 * 60)
'''
