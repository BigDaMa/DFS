from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_sequential_search
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
from fastsklearnfeature.interactiveAutoML.feature_selection.TraceRFECV import TraceRFECV
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

X_train = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_train.data', delimiter=' ', header=None).values[:,0:500] [0:100,:]
y_train = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_train.labels', delimiter=' ', header=None).values [0:100]

print("loaded")

# generate grid
complexity_grid = np.arange(1, X_train.shape[1]+1)
accuracy_grid = np.arange(0.0, 1.0, 1.0 / len(complexity_grid))

print(complexity_grid)
print(accuracy_grid)

grid = list(itertools.product(complexity_grid, accuracy_grid))

print(len(grid))

meta_X_data = np.matrix(grid)


#run 10 random combinations
ids = np.random.choice(len(grid), size=10, replace=False, p=None)

kfold = StratifiedKFold(n_splits=10, shuffle=False)
scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

max_time = 1 * 30

meta_X_train = np.zeros((10, 2))
runtimes = []


for rounds in range(10):
	for i in range(len(ids)):
		complexity = meta_X_data[ids[i], 0]
		accuracy = meta_X_data[ids[i], 1]
		try:
			runtime = run_sequential_search(X_train, y_train, model=LogisticRegression(random_state=42), kfold=copy.deepcopy(kfold), scoring=scoring, forward=True, max_complexity=int(complexity), min_accuracy=accuracy, fit_time_out=max_time)
		except:
			runtime = max_time
		print(runtime)

		if rounds==0:
			meta_X_train[i] = meta_X_data[ids[i]]
		else:
			meta_X_train = np.vstack([meta_X_train, meta_X_data[ids[i]]])
		runtimes.append(runtime)

	al_model = RandomForestRegressor(n_estimators=10)
	al_model.fit(meta_X_train, runtimes)

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

	runtime_predictions = al_model.predict(meta_X_data)

	df = pd.DataFrame.from_dict(np.array([meta_X_data[:, 0].A1, meta_X_data[:, 1].A1, runtime_predictions]).T)
	df.columns = ['Max Complexity', 'Min Accuracy', 'Estimated Runtime']
	pivotted = df.pivot('Max Complexity', 'Min Accuracy', 'Estimated Runtime')
	sns.heatmap(pivotted, cmap='RdBu')
	plt.show()


