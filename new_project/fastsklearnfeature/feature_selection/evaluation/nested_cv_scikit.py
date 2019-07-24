from sklearn.model_selection import GridSearchCV
from fastsklearnfeature.fastfeature_utils.candidate2pipeline import generate_pipeline
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import copy
import numpy as np

from typing import List, Dict, Set
import numpy as np
from fastsklearnfeature.configuration.Config import Config
from fastsklearnfeature.candidates.RawFeature import RawFeature

import tqdm
import multiprocessing as mp
from fastsklearnfeature.feature_selection.evaluation import nested_my_globale_module
import fastsklearnfeature.feature_selection.evaluation.my_globale_module as my_globale_module
import itertools
from sklearn.model_selection import StratifiedKFold

class hashabledict(dict):
	def __hash__(self):
		return hash(tuple(sorted(self.items())))




def gridsearch(X, y, grid_search_parameters, classifier, ids, score, test_ids):
	hyperparam_to_score_list = {}

	my_keys = list(grid_search_parameters.keys())

	test_fold_predictions = {}

	for parameter_combination in itertools.product(*[grid_search_parameters[k] for k in my_keys]):
		parameter_set = hashabledict(zip(my_keys, parameter_combination))
		hyperparam_to_score_list[parameter_set] = []
		test_fold_predictions[parameter_set] = []
		for fold in range(len(ids)):
			clf = classifier
			clf.set_params(**parameter_set)
			clf.fit(X[ids[fold][0]], y[ids[fold][0]])
			hyperparam_to_score_list[parameter_set].append(score(clf, X[ids[fold][1]], y[ids[fold][1]]))

	best_param = None
	best_mean_cross_val_score = -float("inf")
	for parameter_config, score_list in hyperparam_to_score_list.items():
		mean_score = np.mean(score_list)
		if mean_score > best_mean_cross_val_score:
			best_param = parameter_config
			best_mean_cross_val_score = mean_score

	train_and_validation_ids = copy.deepcopy(ids[0][0])
	train_and_validation_ids.extend(ids[0][1])

	# refit to entire training and test on test set
	clf = classifier
	clf.set_params(**best_param)
	clf.fit(X[train_and_validation_ids], y[train_and_validation_ids])
	test_score = score(clf, X[test_ids], y[test_ids])

	return test_score


def run_nested_cross_validation(feature: CandidateFeature, splitted_values_train, splitted_target_train, parameters, model, preprocessed_folds, score):
	try:
		pipeline = generate_pipeline(feature, model)

		# replace parameter keys
		new_parameters = copy.deepcopy(parameters)
		old_keys = list(new_parameters.keys())
		for k in old_keys:
			if not str(k).startswith('c__'):
				new_parameters['c__' + str(k)] = new_parameters.pop(k)

		preprocessed_folds = []
		for train, test in StratifiedKFold(n_splits=20, random_state=42).split(
				splitted_values_train,
				splitted_target_train):
			preprocessed_folds.append((train, test))

		fold_ids = []
		for fold in range(len(preprocessed_folds)):
			fold_ids.append(preprocessed_folds[fold][1])

		nested_cv_scores = []
		for test_fold in range(len(fold_ids)):
			test_ids = fold_ids[test_fold]

			my_set = set(range(len(fold_ids)))
			my_set.remove(test_fold)

			cv_split_ids = []

			for validation_fold in my_set:
				validation_ids = fold_ids[validation_fold]

				new_my_set = copy.deepcopy(my_set)
				new_my_set.remove(validation_fold)

				training_ids = []
				for train_fold in new_my_set:
					training_ids.extend(fold_ids[train_fold])

				cv_split_ids.append((training_ids, validation_ids))

			nested_cv_scores.append(gridsearch(splitted_values_train, splitted_target_train, new_parameters, pipeline, cv_split_ids, score, test_ids))

		return np.average(nested_cv_scores)
	except Exception as e:
		#print(e)
		return 0.0


'''
def run_nested_cross_validation(feature: CandidateFeature, splitted_values_train, splitted_target_train, parameters, model, preprocessed_folds, score):

	try:
		X_train = splitted_values_train
		y_train = splitted_target_train

		X_test = splitted_values_train
		y_test = splitted_target_train

		pipeline = generate_pipeline(feature, model)

		#replace parameter keys

		new_parameters = copy.deepcopy(parameters)
		old_keys = list(new_parameters.keys())
		for k in old_keys:
			if not str(k).startswith('c__'):
				new_parameters['c__' + str(k)] = new_parameters.pop(k)

		cv = GridSearchCV(pipeline, param_grid=new_parameters, scoring=score, cv=40, refit=True)
		cv.fit(X_train, y_train)
		return cv.score(X_test, y_test)
	except:
		return 0.0
'''

def run_nested_cross_validation_global(feature_id: int):
	feature: CandidateFeature = nested_my_globale_module.candidate_list_global[feature_id]

	splitted_values_train = nested_my_globale_module.splitted_values_train
	splitted_target_train = nested_my_globale_module.splitted_target_train

	parameters = my_globale_module.grid_search_parameters_global
	model = my_globale_module.classifier_global
	preprocessed_folds = my_globale_module.preprocessed_folds_global
	score = my_globale_module.score_global

	feature.runtime_properties['nested_cv_score'] = run_nested_cross_validation(feature, splitted_values_train, splitted_target_train, parameters, model, preprocessed_folds, score)
	return feature

def nested_cv_score_parallel(candidates: List[CandidateFeature], splitted_values_train, splitted_target_train,  n_jobs: int = int(Config.get_default("parallelism", mp.cpu_count()))) -> List[CandidateFeature]:
	nested_my_globale_module.candidate_list_global = candidates
	nested_my_globale_module.splitted_values_train = splitted_values_train
	nested_my_globale_module.splitted_target_train = splitted_target_train

	with mp.Pool(processes=n_jobs) as pool:
		my_function = run_nested_cross_validation_global
		candidates_ids = list(range(len(candidates)))

		if Config.get_default("show_progess", 'True') == 'True':
			results = []
			for x in tqdm.tqdm(pool.imap_unordered(my_function, candidates_ids), total=len(candidates_ids)):
				results.append(x)
		else:
			results = pool.map(my_function, candidates_ids)


	return results



