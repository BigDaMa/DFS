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





path = '/tmp/'
#path = '/home/felix/phd/feature_constraints/bestk_promoters/'
#path = '/home/felix/phd/feature_constraints/bestk_experiments_madelon/'
#path = '/home/felix/phd/feature_constraints/bestk_arcene/'
#path = '/home/felix/phd/feature_constraints/bestk_tumor/'


#path = '/home/felix/phd/feature_constraints/data_promoters_knn/'
#path = '/home/felix/phd/feature_constraints/data_promoters_logistic_regression/'
#path = '/home/felix/phd/feature_constraints/data_promoters_dec_tree/'


#path = '/home/felix/phd/feature_constraints/madelon_knn/'
#path = '/home/felix/phd/feature_constraints/cancer_knn/'

file_lists = glob.glob(path + "all*.p")

map_k_to_results = pickle.load(open(file_lists[0], "rb"))
my_size = len(map_k_to_results)
max_complexity = my_size -1#20#my_size -1
space = hp.randint('k', max_complexity)

for f_name in file_lists:
	map_k_to_results = pickle.load(open(f_name, "rb"))

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

			print(len(map_k_to_results))

			ks = []
			runtimes = []
			accuracies = []

			start_time = time.time()

			trials = Trials()
			i = 1
			while True:
				fmin(partial(get_k_score, map_k_to_results=map_k_to_results), space=space, algo=tpe.suggest, max_evals=i, trials=trials)

				runtimes.append(((time.time() - start_time) + sum([r['time'] for r in trials.results]) + map_k_to_results[-1][1]) / 80.0)

				accuracies.append(max([r['test_score'] for r in trials.results]))
				i += 1

				if i > len(map_k_to_results):
					break

			plt.plot(runtimes, accuracies, label=f_name.split('/')[-1][3:].split('.')[0])


plt.legend(loc=(1.04,0))
plt.xlabel('Time (min)')
plt.ylabel('AUC')
plt.subplots_adjust(right=0.7)
plt.show()
