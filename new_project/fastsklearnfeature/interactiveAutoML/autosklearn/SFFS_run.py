import numpy as np
import copy
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import multiprocessing
import pandas as pd
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature

import fastsklearnfeature.interactiveAutoML.autosklearn.mp_global as mp_globalsfs
from multiprocessing import Pool
import tqdm
import time
import pickle

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


def get_model(c):
	return ('clf', LogisticRegression(class_weight='balanced', C=c, solver='sag'))
	#return ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=c))
'''
def get_params():
	return [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	#return [100]
	#return [1.0]
'''

pARAMS = [1.0]

def run_fold(cv):
	mask = mp_globalsfs.mask
	c = mp_globalsfs.parameter
	pipeline = Pipeline([
		('imputation', SimpleImputer()),
		('selection', MaskSelection(mask)),
		get_model(c)
	])

	X_train, y_train, X_test, y_test = mp_globalsfs.data_per_fold[cv]
	pipeline.fit(X_train, pd.DataFrame(y_train))

	return auc_scorer(pipeline, X_test, y_test)


def execute_feature_combo1(feature_combo,feature_combo_id=0, params=pARAMS):
	mask = np.zeros(mp_globalsfs.data_per_fold[0][0].shape[1], dtype=bool)
	for fc in feature_combo:
		mask[fc] = True

	mp_globalsfs.mask = mask

	hyperparameter_search_scores = []
	for c in params:
		pipeline = Pipeline([
			('imputation', SimpleImputer()),
			('selection', MaskSelection(mask)),
			get_model(c)
		])

		mp_globalsfs.parameter = c

		cv_scores = []
		with Pool(processes=multiprocessing.cpu_count()) as p:
			cv_scores = list(tqdm.tqdm(p.imap(run_fold, range(len(mp_globalsfs.data_per_fold))), total=len(mp_globalsfs.data_per_fold)))

		hyperparameter_search_scores.append(np.mean(cv_scores))

	return (feature_combo_id, np.max(hyperparameter_search_scores), params[np.argmax(hyperparameter_search_scores)])

def execute_feature_combo(feature_combo_id):
	feature_combo = mp_globalsfs.feature_combos[feature_combo_id]

	return execute_feature_combo1(feature_combo, feature_combo_id, [0.001, 0.01, 0.1, 1, 10, 100, 1000])
	#return execute_feature_combo1(feature_combo, feature_combo_id, pARAMS)


def get_test_auc(feature_combo, X_train_transformed, y_train, X_test_transformed, y_test, hyperparam):

	mask = np.zeros(mp_globalsfs.data_per_fold[0][0].shape[1], dtype=bool)
	for fc in feature_combo:
		mask[fc] = True

	pipeline = Pipeline([
		('imputation', SimpleImputer()),
		('selection', MaskSelection(mask)),
		get_model(hyperparam)
	])

	pipeline.fit(X_train_transformed, y_train)

	return auc_scorer(pipeline, X_test_transformed, y_test)

def get_test_auc_and_coeff(feature_combo, X_train_transformed, y_train, X_test_transformed, y_test, hyperparam):

	mask = np.zeros(mp_globalsfs.data_per_fold[0][0].shape[1], dtype=bool)
	for fc in feature_combo:
		mask[fc] = True

	pipeline = Pipeline([
		('imputation', SimpleImputer()),
		('selection', MaskSelection(mask)),
		get_model(hyperparam)
	])

	pipeline.fit(X_train_transformed, y_train)

	return auc_scorer(pipeline, X_test_transformed, y_test), pipeline.named_steps['clf'].coef_[0]



def f2str(feature_combo, featurenames):
	my_str = ''
	for f in feature_combo:
		my_str += featurenames[f] + ', '
	my_str = my_str[:-2]
	return my_str

def parallel_run(X_train, y_train, X_test=None, y_test=None, feature_generator=None, folds=3, number_cvs=1):

	start_hpo = time.time()

	pipeline_train = copy.deepcopy(feature_generator.pipeline_)
	X_train_transformed = pipeline_train.fit_transform(X_train)
	X_test_transformed = pipeline_train.transform(X_test)

	current_feature_set = []
	remaining_features = []

	featurenames = []
	for myfi in range(len(feature_generator.numeric_features)):
		featurenames.append(str(feature_generator.numeric_features[myfi]))
		if isinstance(feature_generator.numeric_features[myfi], RawFeature) or isinstance(
				feature_generator.numeric_features[myfi].transformation, OneHotTransformation):
			current_feature_set.append(myfi)
		else:
			remaining_features.append(myfi)

	mp_globalsfs.data_per_fold = []
	for ncvs in range(number_cvs):
		for train, test in StratifiedKFold(n_splits=folds, random_state=42+ncvs).split(
				X_train,
				y_train):
			pipeline_fold = copy.deepcopy(feature_generator.pipeline_)
			X_train_fold = pipeline_fold.fit_transform(X_train[train])
			y_train_fold = y_train[train]

			X_test_fold = pipeline_fold.transform(X_train[test])
			y_test_fold = y_train[test]

			mp_globalsfs.data_per_fold.append((X_train_fold, y_train_fold, X_test_fold, y_test_fold))

	current_feature_set = list(range(len(feature_generator.numeric_features)))

	_, score, my_param = execute_feature_combo1(current_feature_set, feature_combo_id=0, params=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
	test_auc, coefficients = get_test_auc_and_coeff(current_feature_set, X_train_transformed, y_train, X_test_transformed, y_test, my_param)
	print('feature number: ' + str(len(current_feature_set)) + ' cv score:' + str(score) + ' test auc: ' + str(test_auc))

	return time.time() - start_hpo, test_auc






