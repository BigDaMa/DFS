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

from fastsklearnfeature.interactiveAutoML.new_bench import run_search
from fastsklearnfeature.configuration.Config import Config

def my_function(id):
	X_train = my_global_utils1.X_train
	y_train = my_global_utils1.y_train
	data_name = my_global_utils1.data_name
	my_search_strategy = my_global_utils1.my_search_strategy[id]
	max_time = my_global_utils1.max_time
	run_experiments_for_strategy(X_train, y_train, data_name, my_search_strategy, max_time)


'''
my_global_utils1.X_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000][0:1000,:]
my_global_utils1.y_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.labels', delimiter=' ', header=None).values[0:1000]
my_global_utils1.data_name = 'ARCENE_sample1k'
'''

'''
data = pd.read_csv(Config.get('data_path') + '/musk/musk.csv', delimiter=',', header=0)
my_global_utils1.y_train = data['class'].values
my_global_utils1.X_train = data[data.columns.difference(['class', 'ID', 'molecule_name', 'conformation_name'])].values
my_global_utils1.data_name = 'musk'
'''

'''
data = pd.read_csv(Config.get('data_path') + '/promoters/dataset_106_molecular-biology_promoters.csv', delimiter=',', header=0)
my_global_utils1.y_train = data['class'].values
my_global_utils1.X_train = data[data.columns.difference(['class', 'instance'])].values
my_global_utils1.data_name = 'promoters'
'''

'''
data = pd.read_csv(Config.get('data_path') + '/leukemia/leukemia.csv', delimiter=',', header=0)
my_global_utils1.y_train = data['CLASS'].values
my_global_utils1.X_train = data[data.columns.difference(['CLASS'])].values
my_global_utils1.data_name = 'leukemia'
'''


data = pd.read_csv(Config.get('data_path') + '/breastTumor/breastTumor.csv', delimiter=',', header=0)
my_global_utils1.y_train = data['binaryClass'].values
my_global_utils1.X_train = data[data.columns.difference(['binaryClass'])].values
my_global_utils1.data_name = 'breastTumor'



'''

data = pd.read_csv(Config.get('data_path') + '/coil2000/coil2000.csv', delimiter=',', header=0)
my_global_utils1.y_train = data['CARAVAN'].values
my_global_utils1.X_train = data[data.columns.difference(['CARAVAN'])].values
my_global_utils1.data_name = 'coil2000'
'''


'''
my_global_utils1.X_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.data', delimiter=' ', header=None).values[:,0:500] [0:100,:]
my_global_utils1.y_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.labels', delimiter=' ', header=None).values [0:100]
my_global_utils1.data_name = 'madelon_sample'
'''


#my_global_utils1.my_search_strategy = [run_sequential_search, run_hyperopt_search_kbest_info, run_forward_seq_search, run_al_k_search]
my_global_utils1.my_search_strategy = [run_search.run_hyperopt_search_kbest_forest, run_search.run_hyperopt_search_kbest_l1,
									   run_search.run_hyperopt_search_kbest_fcbf, run_search.run_hyperopt_search_kbest_relieff,
									   run_search.run_hyperopt_search_kbest_info, run_search.run_hyperopt_search_kbest_chi2, run_search.run_hyperopt_search_kbest_f_classif, run_search.run_hyperopt_search_kbest_variance,
									   run_search.run_sequential_search, run_search.run_forward_seq_search, run_search.run_al_k_search
									   ]
#my_global_utils1.my_search_strategy = [run_hyperopt_search_kbest_forest]
#my_global_utils1.max_time = 20 * 60
my_global_utils1.max_time = 20 * 60


my_global_utils1.y_train = my_global_utils1.y_train


n_jobs = len(my_global_utils1.my_search_strategy)
with mp.Pool(processes=n_jobs) as pool:
	results = pool.map(my_function, range(len(my_global_utils1.my_search_strategy)))
