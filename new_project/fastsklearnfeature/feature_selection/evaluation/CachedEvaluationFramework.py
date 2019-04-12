from typing import List, Dict, Set
import numpy as np
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.configuration.Config import Config
from fastsklearnfeature.candidate_generation.feature_space.explorekit_transformations import get_transformation_for_feature_space
from fastsklearnfeature.feature_selection.evaluation.EvaluationFramework import EvaluationFramework
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import time
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import itertools
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
import warnings
from functools import partial
import tqdm
import multiprocessing as mp

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))



def grid_search(train_transformed, test_transformed, training_all, one_test_set_transformed,
                grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target, test_target):

    hyperparam_to_score_list = {}

    my_keys = list(grid_search_parameters.keys())

    for parameter_combination in itertools.product(*[grid_search_parameters[k] for k in my_keys]):
        parameter_set = hashabledict(zip(my_keys, parameter_combination))
        hyperparam_to_score_list[parameter_set] = []
        for fold in range(len(train_transformed)):
            clf = classifier(**parameter_set)
            clf.fit(train_transformed[fold], target_train_folds[fold])
            y_pred = clf.predict(test_transformed[fold])
            hyperparam_to_score_list[parameter_set].append(score._sign * score._score_func(target_test_folds[fold], y_pred, **score._kwargs))

    best_param = None
    best_mean_cross_val_score = -float("inf")
    for parameter_config, score_list in hyperparam_to_score_list.items():
        mean_score = np.mean(score_list)
        if mean_score > best_mean_cross_val_score:
            best_param = parameter_config
            best_mean_cross_val_score = mean_score

    test_score = None
    if Config.get_default('score.test', 'False') == 'True':
        # refit to entire training and test on test set
        clf = classifier(**best_param)
        clf.fit(training_all, train_y_all_target)
        y_pred = clf.predict(one_test_set_transformed)
        test_score = score._sign * score._score_func(test_target, y_pred, **score._kwargs)

        #np.save('/tmp/true_predictions', self.test_target)


    return best_mean_cross_val_score, test_score, best_param, y_pred





def evaluate(candidate: CandidateFeature, global_starting_time,
             grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target,
             test_target,
             max_timestamp, preprocessed_folds, epsilon, complexity_delta
             ):

    if type(max_timestamp) != type(None) and time.time() >= max_timestamp:
        raise RuntimeError('Out of time!')

    train_transformed = [None] * len(preprocessed_folds)
    test_transformed = [None] * len(preprocessed_folds)

    #test
    training_all = None
    one_test_set_transformed = None

    if isinstance(candidate, RawFeature):

        if Config.get_default('score.test', 'False') == 'True':
            training_all = candidate.runtime_properties['training_all']
            one_test_set_transformed = candidate.runtime_properties['one_test_set_transformed']

        train_transformed = candidate.runtime_properties['train_transformed']
        test_transformed = candidate.runtime_properties['test_transformed']

    else:

        #print(self.name_to_train_transformed.keys())

        #merge columns from parents
        for fold in range(len(preprocessed_folds)):
            train_transformed_input = np.hstack([p.runtime_properties['train_transformed'][fold] for p in candidate.parents])
            test_transformed_input = np.hstack([p.runtime_properties['test_transformed'][fold] for p in candidate.parents])

            candidate.transformation.fit(train_transformed_input)
            train_transformed[fold] = candidate.transformation.transform(train_transformed_input)
            test_transformed[fold] = candidate.transformation.transform(test_transformed_input)

        if Config.get_default('score.test', 'False') == 'True':
            training_all_input = np.hstack(
                [p.runtime_properties['training_all'] for p in candidate.parents])
            one_test_set_transformed_input = np.hstack(
                [p.runtime_properties['one_test_set_transformed'] for p in candidate.parents])

            candidate.transformation.fit(training_all_input)
            training_all = candidate.transformation.transform(training_all_input)
            one_test_set_transformed = candidate.transformation.transform(one_test_set_transformed_input)

    candidate.runtime_properties['score'], candidate.runtime_properties['test_score'], candidate.runtime_properties['hyperparameters'], y_pred = grid_search(train_transformed, test_transformed, training_all, one_test_set_transformed,
                grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target, test_target)

    if Config.get_default('store.predictions', 'False') == 'True':
        candidate.runtime_properties['predictions'] = y_pred


    #only save the transformed data if we need it in the future
    candidate.runtime_properties['passed'] = False
    if isinstance(candidate, RawFeature) or candidate.runtime_properties['score'] - np.max([p.runtime_properties['score'] for p in candidate.parents]) / complexity_delta > epsilon:
        candidate.runtime_properties['passed'] = True
        if not isinstance(candidate, RawFeature):
            candidate.runtime_properties['train_transformed'] = train_transformed
            candidate.runtime_properties['test_transformed'] = test_transformed

            if Config.get_default('score.test', 'False') == 'True':
                candidate.runtime_properties['training_all'] = training_all
                candidate.runtime_properties['one_test_set_transformed'] = one_test_set_transformed

            # derive properties
            if not isinstance(candidate, RawFeature):
                candidate.derive_properties(candidate.runtime_properties['train_transformed'][0])


        # remove parents' materialization
        candidate.get_name()
        candidate.get_features_from_identity_candidate()
        candidate.get_complexity()
        candidate.parents = None
        return candidate


    return None


def evaluate_catch(candidate: CandidateFeature, global_starting_time,
             grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target,
             test_target,
             max_timestamp, preprocessed_folds, epsilon, complexity_delta
             ):
    result = None
    time_start_gs = time.time()
    try:
        result = evaluate(candidate, global_starting_time, grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target,
             test_target, max_timestamp, preprocessed_folds, epsilon, complexity_delta)
    except Exception as e:
        warnings.warn(str(candidate) + " -> " + str(e), RuntimeWarning)

        '''
        candidate.runtime_properties['exception'] = e
        candidate.runtime_properties['score'] = -1.0
        candidate.runtime_properties['test_score'] = -1.0
        candidate.runtime_properties['hyperparameters'] = {}
        '''

    if type(result) != type(None):
        result.runtime_properties['execution_time'] = time.time() - time_start_gs
        result.runtime_properties['global_time'] = time.time() - global_starting_time

    return result

def evaluate_no_catch(candidate: CandidateFeature, global_starting_time,
             grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target,
             test_target,
             max_timestamp, preprocessed_folds, epsilon, complexity_delta
             ):
    time_start_gs = time.time()

    evaluate(candidate, global_starting_time, grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target,
         test_target, max_timestamp, preprocessed_folds, epsilon, complexity_delta)


    candidate.runtime_properties['execution_time'] = time.time() - time_start_gs
    candidate.runtime_properties['global_time'] = time.time() - global_starting_time

    return candidate

def evaluate_candidates(candidates, global_starting_time, grid_search_parameters, score, classifier, target_train_folds,
                        target_test_folds, train_y_all_target,
             test_target, max_timestamp, preprocessed_folds, epsilon, complexity_delta):
    pool = mp.Pool(processes=int(Config.get_default("parallelism", mp.cpu_count())))

    my_function = partial(evaluate_catch,
                          global_starting_time=global_starting_time,
                          grid_search_parameters=grid_search_parameters,
                          score=score,
                          classifier=classifier,
                          target_train_folds=target_train_folds,
                          target_test_folds=target_test_folds,
                          train_y_all_target=train_y_all_target,
                          test_target=test_target,
                          max_timestamp=max_timestamp,
                          preprocessed_folds=preprocessed_folds,
                          epsilon=epsilon,
                          complexity_delta=complexity_delta
                          )


    if Config.get_default("show_progess", 'True') == 'True':
        results = []
        for x in tqdm.tqdm(pool.imap_unordered(my_function, candidates), total=len(candidates)):
            results.append(x)
    else:
        results = pool.map(my_function, candidates)


    return results


class CachedEvaluationFramework(EvaluationFramework):
    def __init__(self, dataset_config, classifier=LogisticRegression, grid_search_parameters={'penalty': ['l2'],
                                                                                                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'solver': ['lbfgs'],
                                                                                                'class_weight': ['balanced'],
                                                                                                'max_iter': [10000],
                                                                                                'multi_class':['auto']
                                                                                                },
                 transformation_producer=get_transformation_for_feature_space
                 ):
        super(CachedEvaluationFramework, self).__init__(dataset_config, classifier, grid_search_parameters, transformation_producer)


    def generate_target(self):
        current_target = self.dataset.splitted_target['train']

        if isinstance(self.classifier(), ClassifierMixin):
            label_encoder = LabelEncoder()
            label_encoder.fit(current_target)

            current_target = label_encoder.transform(current_target)

            if Config.get_default('score.test', 'False') == 'True':
                self.test_target = label_encoder.transform(self.dataset.splitted_target['test'])
                self.train_y_all_target = label_encoder.transform(self.train_y_all)


            self.preprocessed_folds = []
            for train, test in StratifiedKFold(n_splits=self.folds, random_state=42).split(self.dataset.splitted_values['train'],
                                                                                   current_target):
                self.preprocessed_folds.append((train, test))
        elif isinstance(self.classifier(), RegressorMixin):

            if Config.get_default('score.test', 'False') == 'True':
                self.test_target = self.dataset.splitted_target['test']
                self.train_y_all_target = self.train_y_all

            self.preprocessed_folds = []
            for train, test in KFold(n_splits=self.folds, random_state=42).split(
                    self.dataset.splitted_values['train'],
                    current_target):
                self.preprocessed_folds.append((train, test))
        else:
            pass

        self.target_train_folds = [None] * self.folds
        self.target_test_folds = [None] * self.folds

        for fold in range(len(self.preprocessed_folds)):
            self.target_train_folds[fold] = current_target[self.preprocessed_folds[fold][0]]
            self.target_test_folds[fold] = current_target[self.preprocessed_folds[fold][1]]





















