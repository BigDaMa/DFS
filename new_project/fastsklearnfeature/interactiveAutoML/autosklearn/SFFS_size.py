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
from sklearn.model_selection import train_test_split

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


def get_model(c):
	return ('clf', LogisticRegression(class_weight='balanced', C=c))
	#return ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=c))
'''
def get_params():
	return [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	#return [100]
	#return [1.0]
'''

pARAMS = [1.0]

def execute_feature_combo1(feature_combo,feature_combo_id=0, params = pARAMS):
	mask = np.zeros(mp_globalsfs.data_per_fold[0][0].shape[1], dtype=bool)
	for fc in feature_combo:
		mask[fc] = True

	hyperparameter_search_scores = []
	for c in params:
		pipeline = Pipeline([
			('imputation', SimpleImputer()),
			('selection', MaskSelection(mask)),
			get_model(c)
		])

		cv_scores = []
		for cv in range(len(mp_globalsfs.data_per_fold)):
			X_train, y_train, X_test, y_test = mp_globalsfs.data_per_fold[cv]
			pipeline.fit(X_train, pd.DataFrame(y_train))

			cv_scores.append(auc_scorer(pipeline, X_test, y_test))
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

def parallel_size(X_train, y_train, X_test=None, y_test=None, floating=True, max_number_features=10, feature_generator=None, folds=3, number_cvs=1):
	base_featurenames = []
	for myf in feature_generator.numeric_features:
		base_featurenames.append(str(myf))


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

	current_feature_set = list(range(len(feature_generator.numeric_features)))
	for size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
		X_train_new, _, y_train_new, _ = train_test_split(X_train, y_train, train_size=size, random_state=42, stratify=y_train)

		mp_globalsfs.data_per_fold = []
		for ncvs in range(number_cvs):
			for train, test in StratifiedKFold(n_splits=folds, random_state=42 + ncvs).split(
					X_train_new,
					y_train_new):
				pipeline_fold = copy.deepcopy(feature_generator.pipeline_)
				X_train_fold = pipeline_fold.fit_transform(X_train_new[train])
				y_train_fold = y_train_new[train]

				X_test_fold = pipeline_fold.transform(X_train_new[test])
				y_test_fold = y_train_new[test]

				mp_globalsfs.data_per_fold.append((X_train_fold, y_train_fold, X_test_fold, y_test_fold))


		pipeline_train = copy.deepcopy(feature_generator.pipeline_)
		X_train_transformed = pipeline_train.fit_transform(X_train_new)

		_, score, my_param = execute_feature_combo1(current_feature_set, feature_combo_id=0, params=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
		test_auc, coefficients = get_test_auc_and_coeff(current_feature_set, X_train_transformed, y_train_new, X_test_transformed, y_test, my_param)
		print('size: ' + str(size) + ' cv score:' + str(score) + ' test auc: ' + str(test_auc))

