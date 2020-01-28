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

new_y_train = copy.deepcopy(new_X_train[:, sensitive_ids[0]])
new_X_train[:, sensitive_ids] = 0

ranking_model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=0)
ranking_model.fit(new_X_train, new_y_train)
fairness_ranking = ranking_model.feature_importances_

fairness_ranking[sensitive_ids] = np.max(fairness_ranking)

fairness_ranking *= -1
pickle.dump(fairness_ranking, open("/home/felix/phd/ranking_exeriments/fairness_ranking.p", "wb"))


#ranking by robustness


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(type(y_train))
print(y_train.shape)

X_train_rob, X_test_rob, y_train_rob, y_test_rob = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

print(X_train_rob.shape)

from art.attacks import ZooAttack
from art.attacks import FastGradientMethod

robustness_ranking = np.zeros(X_train_rob.shape[1])

for feature_i in range(X_train_rob.shape[1]):
	feature_ids = list(range(X_train_rob.shape[1]))
	del feature_ids[feature_i]

	from sklearn.svm import LinearSVC
	model = LinearSVC()
	#model = LogisticRegression()#ExtraTreesClassifier(n_estimators=1)
	#model.fit(X_train_rob[:,feature_ids], y_train_rob)
	
	tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	cv = GridSearchCV(LinearSVC(), tuned_parameters)
	cv.fit(X_train_rob[:,feature_ids], y_train_rob)
	model = cv.best_estimator_
	
	classifier = SklearnClassifier(model=model)
	attack = FastGradientMethod(classifier, eps=0.1, batch_size=1)

	X_test_adv = attack.generate(X_test_rob[:,feature_ids])

	diff = model.score(X_test_rob[:, feature_ids], y_test_rob) - model.score(X_test_adv, y_test_rob)
	print(diff)
	robustness_ranking[feature_i] = diff
robustness_ranking *= -1
pickle.dump(robustness_ranking, open("/home/felix/phd/ranking_exeriments/robustness_ranking.p", "wb"))



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

def f_clf1(hps):
	# Assembing pipeline
	weights = [hps['acc_w'], hps['fair_w'], hps['rob_w']]
	#weights = [0.0, 1.0, 0.0]
	rankings = [accuracy_ranking, fairness_ranking, robustness_ranking]

	clf = LogisticRegression()
	if type(privacy_epsilon) != type(None):
		clf = models.LogisticRegression(epsilon=privacy_epsilon) #The smaller the value is, the better privacy protection

	model = Pipeline([
		('selection', WeightedRankingSelection(scores=rankings, weights=weights, k=hps['k'] + 1, names=np.array(names))),
		('clf', clf)
	])

	return model

def get_knn(hps):
	# Assembing pipeline
	weights = [hps['acc_w'], hps['fair_w'], hps['rob_w']]
	#weights = [0.0, 1.0, 0.0]
	rankings = [accuracy_ranking, fairness_ranking, robustness_ranking]

	clf = KNeighborsClassifier(n_neighbors=3)
	#if type(privacy_epsilon) != type(None):
	#	clf = models.LogisticRegression(epsilon=privacy_epsilon)

	model = Pipeline([
		('selection', WeightedRankingSelection(scores=rankings, weights=weights, k=hps['k'] + 1, names=np.array(names))),
		('clf', clf)
	])

	return model

def f_to_min1(hps, X, y, ncv=5):
	print(hps)
	model = f_clf1(hps)

	robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train, y=y_train, feature_selector=model.named_steps['selection'])

	unit_test_instance_id = 0
	unit_test_scorer = make_scorer(unit_test_score, greater_is_better=True, unit_x=X_test[unit_test_instance_id,:], unit_y=y_test[unit_test_instance_id], X=X_train, y=y_train, pipeline=model)

	#turn n cvs in one
	cv_acc = np.mean(cross_val_score(model, X, pd.DataFrame(y), cv=StratifiedKFold(ncv, random_state=42), scoring=auc_scorer))
	cv_fair = 1.0 - np.mean(cross_val_score(model, X, pd.DataFrame(y), cv=StratifiedKFold(ncv, random_state=42), scoring=fair_train))
	cv_robust = 1.0 - np.mean(cross_val_score(model, X, pd.DataFrame(y), cv=StratifiedKFold(ncv, random_state=42), scoring=robust_scorer))

	cv_unit = np.mean(cross_val_score(model, X, pd.DataFrame(y), cv=StratifiedKFold(ncv, random_state=42), scoring=unit_test_scorer))

	#cv_model_gen = (cv_acc + np.mean(cross_val_score(get_knn(hps), X, pd.DataFrame(y), cv=StratifiedKFold(ncv, random_state=42), scoring=auc_scorer))) / 2
	cv_model_gen = 1.0

	loss = 0.0
	if cv_acc >= min_accuracy and cv_fair >= min_fairness and cv_robust >= min_robustness and cv_model_gen >= min_avg_model_accuracy:
		loss = (min_accuracy - cv_acc) + (min_fairness - cv_fair) + (min_robustness - cv_robust) + (min_avg_model_accuracy - cv_model_gen)
	else:
		if cv_fair < min_fairness:
			loss += (min_fairness - cv_fair)
		if cv_acc < min_accuracy:
			loss += (min_accuracy - cv_acc)
		if cv_robust < min_robustness:
			loss += (min_robustness - cv_robust)
		if cv_model_gen < min_avg_model_accuracy:
			loss += (min_avg_model_accuracy - cv_model_gen)

	print("robust: " + str(cv_robust) + " fair: " + str(cv_fair) + " acc: " + str(cv_acc) + 'avg model acc: '+ str(cv_model_gen) +  ' => loss: ' + str(loss))

	return {'loss': loss, 'status': STATUS_OK, 'model': model}



space = {'k': hp.randint('k', max_number_features),
		 'acc_w': hp.lognormal('acc_w', 0, 1),
         'fair_w': hp.lognormal('fair_w', 0, 1),
		 'rob_w': hp.lognormal('rob_w', 0, 1),
		 }


start_time = time.time()

trials = Trials()
i = 1
while True:
	fmin(partial(f_to_min1, X=X_train, y=y_train), space=space, algo=tpe.suggest, max_evals=i, trials=trials)
	if trials.trials[-1]['result']['loss'] < 0.0:
		model = trials.trials[-1]['result']['model']

		robust_scorer_test = make_scorer(robust_score_test, greater_is_better=True, X_train=X_train, y_train=y_train,
										 X_test=X_test, y_test=y_test,
										 feature_selector=model.named_steps['selection'])

		model.fit(X_train, pd.DataFrame(y_train))
		test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
		test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
		test_robust = 1.0 - robust_scorer_test(model, X_test, pd.DataFrame(y_test))

		print('acc: ' + str(test_acc) + ' fair: ' + str(test_fair) + ' robust: ' + str(test_robust))

		if test_acc > min_accuracy and test_fair > min_fairness and test_robust > min_robustness:
			break

	i += 1


print("time until constraint: " + str(time.time() - start_time))
