from fastsklearnfeature.interactiveAutoML.new_bench.create_runtime_chart_per_search import run_experiments_for_strategy

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
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.ALSelectionK import ALSelectionK

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
from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1

from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_sequential_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_forward_seq_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_al_k_search


def my_function(id):
	X_train = my_global_utils1.X_train
	y_train = my_global_utils1.y_train
	data_name = my_global_utils1.data_name
	my_search_strategy = my_global_utils1.my_search_strategy[id]
	max_time = my_global_utils1.max_time
	run_experiments_for_strategy(X_train, y_train, data_name, my_search_strategy, max_time)


my_global_utils1.X_train = pd.read_csv('/home/felix/phd/feature_constraints/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000]
my_global_utils1.y_train = pd.read_csv('/home/felix/phd/feature_constraints/ARCENE/arcene_train.labels', delimiter=' ', header=None).values
my_global_utils1.data_name = 'ARCENE'

'''
my_global_utils1.X_train = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_train.data', delimiter=' ', header=None).values[:,0:500] [0:100,:]
my_global_utils1.y_train = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_train.labels', delimiter=' ', header=None).values [0:100]
my_global_utils1.data_name = 'madelon_sample'
'''

my_global_utils1.my_search_strategy = [run_sequential_search, run_hyperopt_search, run_forward_seq_search, run_al_k_search]


n_jobs = len(my_global_utils1.my_search_strategy)
with mp.Pool(processes=n_jobs) as pool:
	results = pool.map(my_function, range(len(my_global_utils1.my_search_strategy)))
