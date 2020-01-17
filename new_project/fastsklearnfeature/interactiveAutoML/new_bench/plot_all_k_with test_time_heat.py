import matplotlib.pyplot as plt
import pickle



import glob
import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
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
from sklearn.neighbors import KNeighborsClassifier
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
import pandas as pd
import time


from fastsklearnfeature.interactiveAutoML.feature_selection.RunAllKBestSelection import RunAllKBestSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier

from fastsklearnfeature.configuration.Config import Config

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler
import fastsklearnfeature.interactiveAutoML.feature_selection.WrapperBestK as wrap
from sklearn.ensemble import ExtraTreesClassifier
import time







def get_k_score(hps, map_k_to_results=None):

	return {'loss': -map_k_to_results[hps+1][0],
			'status': STATUS_OK, 'test_score': map_k_to_results[hps+1][2],
			'time': map_k_to_results[hps+1][1]}





#path = '/tmp/'
#path = '/home/felix/phd/feature_constraints/bestk_promoters/'
#path = '/home/felix/phd/feature_constraints/bestk_experiments_madelon/'
#path = '/home/felix/phd/feature_constraints/bestk_arcene/'
#path = '/home/felix/phd/feature_constraints/bestk_tumor/'


#path = '/home/felix/phd/feature_constraints/data_promoters_knn/'
#path = '/home/felix/phd/feature_constraints/data_promoters_logistic_regression/'
#path = '/home/felix/phd/feature_constraints/data_promoters_dec_tree/'


path = '/home/felix/phd/feature_constraints/madelon_knn/'
#path = '/home/felix/phd/feature_constraints/cancer_knn/'

#path = '/home/felix/phd/feature_constraints/promoter_knn/'

file_lists = glob.glob(path + "all*.p")

map_k_to_results = pickle.load(open(file_lists[0], "rb"))
my_size = len(map_k_to_results)
max_complexity = my_size -1#20#my_size -1

#accuracy_range = np.arange(0.85, 1.0, (1.0 - 0.85) / 10)
#accuracy_range = np.arange(0.45, 0.62, (0.62 - 0.45) / 10)
#accuracy_range = np.arange(0.6, 0.92, (0.92 - 0.6) / 10)
accuracy_range = np.arange(0.89, 0.93, (0.93 - 0.89) / 10)
complexity_range = range(1, len(map_k_to_results), int(len(map_k_to_results)/float(10)))

print(accuracy_range)
print(list(complexity_range))


runtime_s_list = []
fname_list = []

for f_name in file_lists:
	map_k_to_results = pickle.load(open(f_name, "rb"))

	runtimes = np.ones((len(accuracy_range), len(complexity_range))) * np.inf

	print('\n\n' + f_name)

	if 'allforward.p' in f_name:
		ks = []
		runtimes = []
		accuracies = []

		for k in range(1, len(map_k_to_results)):
			runtimes.append(map_k_to_results[k][1])
			accuracies.append(map_k_to_results[k][2])

		plt.plot(runtimes, accuracies, label=f_name.split('/')[-1][3:].split('.')[0])


	else:

		if len(map_k_to_results) == my_size:

			for accuracy_i in range(len(accuracy_range)):
				for complexity_i in range(len(complexity_range)):
					max_complexity = complexity_range[complexity_i]
					space = hp.randint('k', max_complexity)

					my_runtimes = []
					all_runs = 10
					for runs in range(all_runs):

						start_time = time.time()
						trials = Trials()
						i = 1
						while True:
							fmin(partial(get_k_score, map_k_to_results=map_k_to_results), space=space, algo=tpe.suggest, max_evals=i, trials=trials)

							runtime = (((time.time() - start_time) + sum([r['time'] for r in trials.results]) + map_k_to_results[-1][1]) / 60.0)

							current_acc = max([r['test_score'] for r in trials.results])
							i += 1

							if current_acc >= accuracy_range[accuracy_i]:
								#runtimes[accuracy_i, complexity_i] = runtime
								my_runtimes.append(runtime)
								break

							if i > max_complexity:
								break

					if len(my_runtimes) == all_runs:
						runtimes[accuracy_i, complexity_i] = np.mean(my_runtimes)

			my_name = f_name.split('/')[-1][3:].split('.')[0]
			runtime_s_list.append(copy.deepcopy(runtimes))
			fname_list.append(my_name)


my_tensor = np.array(runtime_s_list)
runtimes = np.ones((len(accuracy_range), len(complexity_range))) * (my_tensor[np.isfinite(my_tensor)].max() + 1.0)
runtime_s_list.append(runtimes)
fname_list.append("not reached by anyone")

my_tensor = np.array(runtime_s_list)

for i in range(len(runtime_s_list)):
	print(fname_list[i])
	print(runtime_s_list[i])
	print('-------------------------\n\n')

print(fname_list)
print(np.argmin(my_tensor, axis=0))
print(np.min(my_tensor, axis=0))

