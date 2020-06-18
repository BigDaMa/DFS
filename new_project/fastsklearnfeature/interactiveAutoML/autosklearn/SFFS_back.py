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

def f2str(feature_combo, featurenames):
	my_str = ''
	for f in feature_combo:
		my_str += featurenames[f] + ', '
	my_str = my_str[:-2]
	return my_str

def parallel_back(X_train, y_train, X_test=None, y_test=None, floating=True, max_number_features=10, feature_generator=None, folds=3, number_cvs=1):
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

	base_features = copy.deepcopy(current_feature_set)

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


	best_score = -1
	best_feature_combination = None


	history = {}

	current_feature_set = list(range(len(feature_generator.numeric_features)))
	_, score, my_param = execute_feature_combo1(current_feature_set, feature_combo_id=0,
												params=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
	test_auc = get_test_auc(current_feature_set, X_train_transformed, y_train, X_test_transformed, y_test, my_param)
	print('all features cv auc: ' + str(score) + ' test auc: ' + str(test_auc))

	pARAMS = [my_param]

	best_score = score
	best_feature_combination = current_feature_set


	'''
	#get intial hyperparameter
	_, score, my_param = execute_feature_combo1(current_feature_set, feature_combo_id=0, params=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
	pARAMS =[my_param]
	history[frozenset(current_feature_set)] = score
	

	test_auc = get_test_auc(current_feature_set, X_train_transformed, y_train, X_test_transformed, y_test, my_param)
	print(f2str(current_feature_set, featurenames, base_featurenames) + ' cv auc: ' + str(
		history[frozenset(current_feature_set)]) + ' test auc: ' + str(test_auc))
	'''


	best_score = score
	best_feature_combination = current_feature_set

	best_accuracy = -1

	mycount= 0
	while (mycount <= max_number_features):



		if floating:
			# select worst feature
			while True:

				new_feature_combos = []
				features_removed = []
				for i in range(len(current_feature_set) - 1, 0, -1):
					new_feature = current_feature_set[i]

					#if new_feature in base_features:
					#	continue

					feature_combo = copy.deepcopy(current_feature_set)
					feature_combo.remove(new_feature)

					# book-keeping to avoid infinite loops
					if frozenset(feature_combo) in history or len(feature_combo) == 0:
						continue
					new_feature_combos.append(feature_combo)
					features_removed.append(new_feature)

				mp_globalsfs.feature_combos = new_feature_combos


				r = []
				with Pool(processes=multiprocessing.cpu_count()) as p:
					r = list(tqdm.tqdm(p.imap(execute_feature_combo, range(len(new_feature_combos))), total=len(new_feature_combos)))

				best_feature_id = -1
				best_accuracy_new = -1
				best_hyperparam_new = None
				for my_result in r:
					feature_combo_id, cv_score, hyperparam = my_result
					feature_combo = mp_globalsfs.feature_combos[feature_combo_id]
					history[frozenset(feature_combo)] = cv_score
					#print(f2str(feature_combo, featurenames) + ': ' + str(cv_score))

					if cv_score > best_accuracy_new:
						best_feature_id = features_removed[feature_combo_id]
						best_accuracy_new = cv_score
						best_hyperparam_new = hyperparam

					if cv_score > best_score:
						best_score = cv_score
						best_feature_combination = feature_combo

				if best_accuracy_new < best_accuracy or best_feature_id == -1:
					break
				else:
					best_accuracy = best_accuracy_new
					mycount += 1

					current_feature_set.remove(best_feature_id)
					remaining_features.append(best_feature_id)

					test_auc = get_test_auc(current_feature_set, X_train_transformed, y_train, X_test_transformed,
											y_test,best_hyperparam_new)
					print(f2str(current_feature_set, featurenames) + ' cv auc: ' + str(
						history[frozenset(current_feature_set)]) + ' test auc: ' + str(test_auc))

	mask = np.zeros(X_train_transformed.shape[1], dtype=bool)
	for fc in best_feature_combination:
		mask[fc] = True

	mask_selection = MaskSelection(mask)
	X_train_new = mask_selection.fit_transform(X_train_transformed)
	X_test_new = mask_selection.transform(X_test_transformed)

	return X_train_new, X_test_new

