from sklearn.model_selection import GridSearchCV
from fastsklearnfeature.fastfeature_utils.candidate2pipeline import generate_pipeline
from fastsklearnfeature.fastfeature_utils.candidate2pipeline import generate_smote_pipeline
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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import operator
from sklearn.metrics import make_scorer

class hashabledict(dict):
	def __hash__(self):
		return hash(tuple(sorted(self.items())))



def customAICc(y, p, k=0):
	resid = y - p
	sse = sum(resid ** 2)

	n = float(len(y))

	AIC = 2 * k + n * np.log(sse / n)
	AICc = AIC + ((2 * k * (k + 1)) / (n - k - 1))

	return AICc





def run_multiple_cross_validation(feature: CandidateFeature, splitted_values_train, splitted_target_train, parameters, model, score):

	#try:
		X_train = splitted_values_train
		y_train = splitted_target_train

		pipeline = generate_pipeline(feature, model)
		#pipeline = generate_smote_pipeline(feature, model)

		multiple_cv_score = []

		multiple_cv_complexity_score = []

		hyperparameters2count = {}

		#print(str(feature) + ' before: ' + str(feature.runtime_properties['hyperparameters']))

		for m_i in range(len(nested_my_globale_module.model_seeds)):
			preprocessed_folds = []
			for train, test in StratifiedKFold(n_splits=len(nested_my_globale_module.splitting_seeds), shuffle=True, random_state=nested_my_globale_module.splitting_seeds[m_i]).split(
					splitted_values_train,
					splitted_target_train):
				preprocessed_folds.append((train, test))


			#replace parameter keys
			new_parameters = copy.deepcopy(parameters)
			new_parameters['random_state'] = [int(nested_my_globale_module.model_seeds[m_i])]
			old_keys = list(new_parameters.keys())
			for k in old_keys:
				if not str(k).startswith('c__'):
					new_parameters['c__' + str(k)] = new_parameters.pop(k)

			scoring = {'accuracy': score, 'complexity': make_scorer(customAICc, greater_is_better=False, needs_proba=True, k=feature.get_complexity())}

			cv = GridSearchCV(pipeline, param_grid=new_parameters, scoring=scoring, cv=preprocessed_folds, refit='accuracy')
			cv.fit(X_train, y_train)
			multiple_cv_score.append(cv.best_score_)

			multiple_cv_complexity_score.append(cv.cv_results_['mean_test_complexity'][cv.best_index_])

			if not hashabledict(cv.best_params_) in hyperparameters2count:
				hyperparameters2count[hashabledict(cv.best_params_)] = 0
			hyperparameters2count[hashabledict(cv.best_params_)] += 1


			'''
			new_parameters = copy.deepcopy(feature.runtime_properties['hyperparameters'])
			new_parameters['random_state'] = int(nested_my_globale_module.model_seeds[m_i])
			old_keys = list(new_parameters.keys())
			for k in old_keys:
				if not str(k).startswith('c__'):
					new_parameters['c__' + str(k)] = new_parameters.pop(k)

			pipeline.set_params(**new_parameters)

			cv_results = cross_validate(pipeline, X_train, y_train, scoring=score, cv=preprocessed_folds)
			multiple_cv_score.append(np.mean(cv_results['test_score']))
			'''


		feature.runtime_properties['hyperparameters'] = max(hyperparameters2count.items(), key=operator.itemgetter(1))[0]

		new_parameters = copy.deepcopy(feature.runtime_properties['hyperparameters'])
		old_keys = list(new_parameters.keys())
		for k in old_keys:
			if str(k).startswith('c__'):
				new_parameters[str(k[3:])] = new_parameters.pop(k)
		feature.runtime_properties['hyperparameters'] = new_parameters


		print(str(feature) + ' AICc: ' + str(np.mean(multiple_cv_complexity_score)))


		#print(str(feature) + ' after: ' + str(feature.runtime_properties['hyperparameters']))

		return np.mean(multiple_cv_score), np.std(multiple_cv_score)
	#except:
	#	return 0.0, 0.0


def run_multiple_cross_validation_global(feature_id: int):
	feature: CandidateFeature = nested_my_globale_module.candidate_list_global[feature_id]

	splitted_values_train = nested_my_globale_module.splitted_values_train
	splitted_target_train = nested_my_globale_module.splitted_target_train

	parameters = my_globale_module.grid_search_parameters_global
	model = my_globale_module.classifier_global
	score = my_globale_module.score_global

	feature.runtime_properties['multiple_cv_score'], feature.runtime_properties['multiple_cv_score_std']  = run_multiple_cross_validation(feature, splitted_values_train, splitted_target_train, parameters, model, score)
	#feature.runtime_properties['score'] = feature.runtime_properties['multiple_cv_score']

	return feature

def multiple_cv_score_parallel(candidates: List[CandidateFeature], splitted_values_train, splitted_target_train,  n_jobs: int = int(Config.get_default("parallelism", mp.cpu_count()))) -> List[CandidateFeature]:
	nested_my_globale_module.candidate_list_global = candidates
	nested_my_globale_module.splitted_values_train = splitted_values_train
	nested_my_globale_module.splitted_target_train = splitted_target_train

	with mp.Pool(processes=n_jobs) as pool:
		my_function = run_multiple_cross_validation_global
		candidates_ids = list(range(len(candidates)))

		if Config.get_default("show_progess", 'True') == 'True':
			results = []
			for x in tqdm.tqdm(pool.imap_unordered(my_function, candidates_ids), total=len(candidates_ids)):
				results.append(x)
		else:
			results = pool.map(my_function, candidates_ids)


	return results



