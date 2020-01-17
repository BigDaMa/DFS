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
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
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
from hyperopt.fmin import generate_trials_to_calculate

'''
data = pd.read_csv(Config.get('data_path') + '/breastTumor/breastTumor.csv', delimiter=',', header=0)
y = data['binaryClass'].values
X = data[data.columns.difference(['binaryClass'])].values
data_name = 'breastTumor'
one_hot = True
'''


data = pd.read_csv(Config.get('data_path') + '/promoters/dataset_106_molecular-biology_promoters.csv', delimiter=',', header=0)
y = data['class'].values
X = data[data.columns.difference(['class', 'instance'])].values
data_name = 'promoters'
one_hot = True




'''
X_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.data', delimiter=' ', header=None).values[:,0:500]
y_train = pd.read_csv(Config.get('data_path') + '/madelon/madelon_train.labels', delimiter=' ', header=None).values
data_name = 'madelon'
one_hot = False
'''

'''
X_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000]
y_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.labels', delimiter=' ', header=None).values
data_name = 'ARCENE'
one_hot = False
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

xshape = X_train.shape[1]
if one_hot:
	encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
	X_train = encoder.fit_transform(X_train)
	xshape = X_train.shape[1]
	X_test = encoder.transform(X_test)

def bindFunction1(estimator):
	def func1(X,y):
		return model_score(X, y, estimator=estimator)
	func1.__name__ = 'score_model_' + estimator.__class__.__name__
	return func1


max_complexity = 20
min_accuracy = 0.97

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


def f_clf1(mask):
	# Assembing pipeline
	model = Pipeline([
		('scale', MinMaxScaler()),
		('selection', MaskSelection(mask)),
		('clf', LogisticRegression())
	])

	return model


def f_to_min1(hps, X, y, ncv=5):
	print(hps)

	mask = np.zeros(X.shape[1], dtype=bool)
	for k, v in hps.items():
		mask[v] = True

	model = f_clf1(mask)
	cv_res = cross_val_score(model, X, pd.DataFrame(y), cv=StratifiedKFold(ncv, random_state=42), scoring=auc_scorer)
	print('result: ' + str(-cv_res.mean()))
	return {'loss': -cv_res.mean(), 'status': STATUS_OK, 'model': model}



#TODO: use probabilities for selection type

space = {}
for f_i in range(max_complexity):
	space['f_' + str(f_i)] = hp.randint('f_' + str(f_i), X_train.shape[1])


start_time = time.time()

trials = Trials()
i = 1
while True:
	fmin(partial(f_to_min1, X=X_train, y=y_train), space=space, algo=tpe.suggest, max_evals=i, trials=trials)
	if trials.best_trial['result']['loss'] * -1 >= min_accuracy:
		model = trials.best_trial['result']['model']
		model.fit(X_train, y_train)
		test_score = auc_scorer(model, X_test, y_test)
		print("test score: " + str(test_score))
		if test_score >= min_accuracy:
			break

	i += 1

print("time until constraint: " + str(time.time() - start_time))