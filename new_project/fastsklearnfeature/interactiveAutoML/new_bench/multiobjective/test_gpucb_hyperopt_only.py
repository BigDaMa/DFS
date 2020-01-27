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
from sklearn.gaussian_process import GaussianProcessRegressor

from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.metrics import loss_sensitivity
from art.metrics import empirical_robustness
from sklearn.pipeline import FeatureUnion


from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from art.classifiers import XGBoostClassifier, LightGBMClassifier, SklearnClassifier
from art.attacks import HopSkipJump


from fastsklearnfeature.interactiveAutoML.feature_selection.RunAllKBestSelection import RunAllKBestSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import unit_test_score

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test

from fastsklearnfeature.configuration.Config import Config

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score

import diffprivlib.models as models

sensitive_attribute = "sex"

n_estimators = 5

df = pd.read_csv(Config.get('data_path') + '/adult/dataset_183_adult.csv', delimiter=',', header=0)
y = df['class']
del df['class']
X = df
one_hot = True

limit = 1000

X_train, X_test, y_train, y_test = train_test_split(X.values[0:limit,:], y.values[0:limit], test_size=0.5, random_state=42)

continuous_columns = [0, 2, 4, 10, 11, 12]

xshape = X_train.shape[1]
if one_hot:
	ct = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False), [1,3,5,6,7,8,9,13])])
	scale = ColumnTransformer([("scale", MinMaxScaler(), continuous_columns)])

	pipeline = FeatureUnion([("o", ct),("s", scale)])

	X_train = pipeline.fit_transform(X_train)
	xshape = X_train.shape[1]
	print(xshape)
	X_test = pipeline.transform(X_test)

	print(ct.get_feature_names())


names = ct.get_feature_names()
for c in continuous_columns:
	names.append(str(X.columns[c]))


pickle.dump(names, open("/home/felix/phd/ranking_exeriments/names.p", "wb"))

print(np.array(names))



#ranking by accuracy
ranking_model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=0)
ranking_model.fit(X_train, y_train)
accuracy_ranking = ranking_model.feature_importances_
pickle.dump(accuracy_ranking, open("/home/felix/phd/ranking_exeriments/accuracy_ranking.p", "wb"))

#ranking by fairness
new_X_train = copy.deepcopy(X_train)

sensitive_ids = []
all_names = ct.get_feature_names()
for fname_i in range(len(all_names)):
	if all_names[fname_i].startswith('onehot__x6' + '_'):
		sensitive_ids.append(fname_i)


print(sensitive_ids)

#ranking by robustness


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(type(y_train))
print(y_train.shape)





## run hyperparameter weighting

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
fair_test = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_test[:, sensitive_ids[0]])


model = LogisticRegression()
model.fit(X_train, y_train)

print("auc: " + str(auc_scorer(model, X_test, y_test)))
print("fair: " + str(fair_train(model, X_train,  pd.DataFrame(y_train))))



from fastsklearnfeature.interactiveAutoML.feature_selection.WeightedRankingSelection import WeightedRankingSelection

'''
weights = [0.0, -1.0, 0.0]
rankings = [accuracy_ranking, fairness_ranking, robustness_ranking]
model = Pipeline([
		('selection', WeightedRankingSelection(scores=rankings, weights=weights, k=1)),
		('clf', LogisticRegression())
	])

cv_acc = np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(5, random_state=42), scoring=auc_scorer))
cv_fair = 1.0 - np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(5, random_state=42), scoring=fair_train))

print(cv_acc)
print(cv_fair)

'''


min_accuracy = 0.80
min_fairness = 0.90
min_robustness = 0.80
privacy_epsilon = None
max_number_features = X_train.shape[1]


min_avg_model_accuracy = 0.0 #does not really make sense

def objective(features):
	model = Pipeline([
		('selection', MaskSelection(features)),
		('clf', LogisticRegression())
	])

	robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train, y=y_train,
								feature_selector=model.named_steps['selection'])
	robust_scorer_test = make_scorer(robust_score_test, greater_is_better=True, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
								feature_selector=model.named_steps['selection'])

	ncv = 5

	cv_acc = np.mean(
		cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=auc_scorer))
	cv_fair = 1.0 - np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=fair_train))
	cv_robust = 1.0 - np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=robust_scorer))
	#cv_robust = 1.0

	print('cv acc: ' + str(cv_acc) + ' cv fair: ' + str(cv_fair) + ' cv robust: ' + str(cv_robust))

	'''
	if cv_acc > min_accuracy and cv_fair > min_fairness and cv_robust > min_robustness:
		model.fit(X_train, pd.DataFrame(y_train))
		test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
		test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
		test_robust = 1.0 - robust_scorer_test(model, X_test, pd.DataFrame(y_test))

		print('acc: ' + str(test_acc) + ' fair: ' + str(test_fair) + ' robust: ' + str(test_robust))

		if test_acc > min_accuracy and test_fair > min_fairness and test_robust > min_robustness:
			my_global_variable.global_check = True

			print("selected features: " + str(np.array(names)[features]))
	'''
	simplicity = -1 * np.sum(features)

	#change objectives
	#cv_acc = 1.0

	#return [cv_acc, cv_fair, cv_robust, simplicity]
	return cv_acc

from sklearn.ensemble import RandomForestRegressor
numberTrees = 10
model = None

def f_to_min1(hps, beta=0.0):
	mask = np.zeros(len(hps), dtype=bool)
	for k, v in hps.items():
		mask[int(k.split('_')[1])] = v

	loss = -1 * objective(mask)

	return {'loss': loss, 'status': STATUS_OK}







def valsToMaskLast(trials):
	mask = np.zeros(len(trials.vals), dtype=bool)
	for k, v in trials.vals.items():
		mask[int(k.split('_')[1])] = v[-1]
	return mask

def valsToMask(trials):
	mask = np.zeros(len(trials.argmin), dtype=bool)
	for k, v in trials.argmin.items():
		mask[int(k.split('_')[1])] = v
	return mask

my_history = []
time_history = []


for runs in range(10):
	start_time = time.time()

	y_train_gp = []
	time_hist = []

	trials = Trials()
	space = {}
	for f_i in range(X_train.shape[1]):
		space['f_' + str(f_i)] = hp.randint('f_' + str(f_i), 2)
	i = 1
	while True:
		fmin(f_to_min1, space=space, algo=tpe.suggest, max_evals=i, trials=trials)
		if i < 53:
			y_new = trials.trials[-1]['result']['loss'] *-1
			time_hist.append(time.time() - start_time)

			y_train_gp.append(y_new)
		else:
			break
		i += 1





	print("time until constraint: " + str(time.time() - start_time))

	print(y_train_gp)

	cummulative_hist = []
	max_acc = -1
	for i in range(len(y_train_gp)):
		if y_train_gp[i] > max_acc:
			max_acc = y_train_gp[i]
		cummulative_hist.append(max_acc)

	print(cummulative_hist)

	my_history.append(np.array(cummulative_hist))
	time_history.append(np.array(time_hist))



print('cumulative_average= ' + str(np.mean(np.array(my_history), axis=0).tolist()))
print('cumulative_std= ' + str(np.std(np.array(my_history), axis=0).tolist()))

print('time_average= ' + str(np.mean(np.array(time_history), axis=0).tolist()))
print('time_std= ' + str(np.std(np.array(time_history), axis=0).tolist()))

'''
current_ids = [-1] * len(my_history)
all_times = np.sort(np.unique(np.array(time_history).flatten()))
all_avg = []
all_std = []

for cu_time in all_times:
	cu_value = []
	for run in range(len(my_history)):
		while current_ids[run]+1 < len(time_history[run]) and time_history[run][current_ids[run]+1] <= cu_time:
			current_ids[run] += 1
		if current_ids[run] >= 0:
			cu_value.append(my_history[run][current_ids[run]])
		else:
			cu_value.append(0.0)
	all_avg.append(np.mean(cu_value))
	all_std.append(np.std(cu_value))

print("times= " + str(all_times.tolist()))
print("averages= " + str(all_avg))
print("std_dev= " + str(all_std))
'''