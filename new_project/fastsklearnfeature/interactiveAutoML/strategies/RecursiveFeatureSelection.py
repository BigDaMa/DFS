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
import xgboost as xgb
#from skrebate import SURF
from skrebate import ReliefF
from sklearn.feature_selection import RFE

which_experiment = 'experiment3'#'experiment1'

numeric_representations: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/features.p", "rb"))

filtered = numeric_representations

'''
filtered = []
for f in numeric_representations:
	if isinstance(f, RawFeature):
		filtered.append(f)
	else:
		if isinstance(f.transformation, OneHotTransformation):
			filtered.append(f)
numeric_representations = filtered
'''

y_test = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_test.p", "rb")).values
#print(y_test)

my_names: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/names.p", "rb"))
print(my_names)

X_train = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/X_train.p", "rb"))
y_train = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_train.p", "rb"))



all_features = CandidateFeature(IdentityTransformation(-1), numeric_representations)
all_standardized = CandidateFeature(MinMaxScalingTransformation(), [all_features])

foreigner = np.array(X_train[:,7])
gender = np.array(['female' in personal_status for personal_status in X_train[:,15]])

my_runner = Runner(c=1.0, sensitive=gender, labels=['bad', 'good'])
#my_runner = Runner(c=1.0, sensitive=foreigner, labels=['bad', 'good'])

history = []

estimator = LogisticRegression()
my_pipeline = Pipeline([('f', all_standardized.pipeline),
						('c', RFE(estimator, 10, step=1))
						])

my_pipeline.fit(X_train, y_train.values)


print(my_pipeline.named_steps['c'].ranking_)

sorted_ids = np.argsort(my_pipeline.named_steps['c'].ranking_)

for c in range(1, len(numeric_representations)+1):
	features = np.zeros(len(numeric_representations))

	features[sorted_ids[0:c]] = True

	if np.sum(features) > 0:
		results = my_runner.run_pipeline(features, runs=1)
		history.append(copy.deepcopy(results))
		print("cv: " + str(results['auc']) + ' test: ' + str(results['test_auc']) + ' fair: ' + str(results['fair']) + ' complexity: ' + str(results['complexity']))

		pickle.dump(history, open("/tmp/recursive_feature_selection1.p", "wb"))