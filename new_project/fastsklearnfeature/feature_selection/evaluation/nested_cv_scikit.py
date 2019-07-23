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

'''
def run_nested_cross_validation(feature: CandidateFeature, splitted_values_train, splitted_target_train, parameters, model, preprocessed_folds, score):

	try:
		nested_cv_scores = []
		for fold in range(len(preprocessed_folds)):
			X_train = splitted_values_train[preprocessed_folds[fold][0]]
			y_train = splitted_target_train[preprocessed_folds[fold][0]]

			X_test = splitted_values_train[preprocessed_folds[fold][1]]
			y_test = splitted_target_train[preprocessed_folds[fold][1]]

			pipeline = generate_pipeline(feature, model)

			#replace parameter keys

			new_parameters = copy.deepcopy(parameters)
			old_keys = list(new_parameters.keys())
			for k in old_keys:
				if not str(k).startswith('c__'):
					new_parameters['c__' + str(k)] = new_parameters.pop(k)

			cv = GridSearchCV(pipeline, param_grid=new_parameters, scoring=score, cv=9, refit=True)
			cv.fit(X_train, y_train)
			nested_cv_scores.append(cv.score(X_test, y_test))
		return np.average(nested_cv_scores)
	except:
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

		cv = GridSearchCV(pipeline, param_grid=new_parameters, scoring=score, cv=20, refit=True)
		cv.fit(X_train, y_train)
		return cv.score(X_test, y_test)
	except:
		return 0.0


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



