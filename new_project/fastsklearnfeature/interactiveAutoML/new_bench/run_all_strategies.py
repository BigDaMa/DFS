
import pandas as pd

import multiprocessing as mp
from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1


from fastsklearnfeature.interactiveAutoML.new_bench.create_runtime_chart_per_search import run_experiments_for_strategy
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_forest
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_l1
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_fcbf
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_relieff
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_info
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_chi2
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_f_classif
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_hyperopt_search_kbest_variance
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_sequential_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_forward_seq_search
from fastsklearnfeature.interactiveAutoML.new_bench.run_search import run_al_k_search
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

'''
data = pd.read_csv(Config.get('data_path') + '/breastTumor/breastTumor.csv', delimiter=',', header=0)
my_global_utils1.y_train = data['binaryClass'].values
my_global_utils1.X_train = data[data.columns.difference(['binaryClass'])].values
my_global_utils1.data_name = 'breastTumor'
'''


'''

data = pd.read_csv(Config.get('data_path') + '/coil2000/coil2000.csv', delimiter=',', header=0)
my_global_utils1.y_train = data['CARAVAN'].values
my_global_utils1.X_train = data[data.columns.difference(['CARAVAN'])].values
my_global_utils1.data_name = 'coil2000'
'''



my_global_utils1.X_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.data', delimiter=' ', header=None).values[:,0:500]
my_global_utils1.y_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.labels', delimiter=' ', header=None).values
my_global_utils1.data_name = 'madelon'



#my_global_utils1.my_search_strategy = [run_sequential_search, run_hyperopt_search_kbest_info, run_forward_seq_search, run_al_k_search]
my_global_utils1.my_search_strategy = [run_hyperopt_search_kbest_forest, run_hyperopt_search_kbest_l1,
									   run_hyperopt_search_kbest_fcbf, run_hyperopt_search_kbest_relieff,
									   run_hyperopt_search_kbest_info, run_hyperopt_search_kbest_chi2, run_hyperopt_search_kbest_f_classif, run_hyperopt_search_kbest_variance,
									   run_sequential_search, run_forward_seq_search, run_al_k_search
									  ]

#my_global_utils1.my_search_strategy = [run_hyperopt_search_kbest_l1]

#my_global_utils1.my_search_strategy = [run_hyperopt_search_kbest_forest]
#my_global_utils1.max_time = 20 * 60
my_global_utils1.max_time = 30 * 60


my_global_utils1.y_train = my_global_utils1.y_train


n_jobs = len(my_global_utils1.my_search_strategy)
with mp.Pool(processes=n_jobs) as pool:
	results = pool.map(my_function, range(len(my_global_utils1.my_search_strategy)))
