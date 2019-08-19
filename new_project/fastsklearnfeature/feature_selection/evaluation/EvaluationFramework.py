from typing import List, Dict, Set
import numpy as np
from fastsklearnfeature.reader.Reader import Reader
from fastsklearnfeature.splitting.Splitter import Splitter
from fastsklearnfeature.splitting.RandomSplitter import RandomSplitter
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from fastsklearnfeature.configuration.Config import Config
from sklearn.pipeline import FeatureUnion
from fastsklearnfeature.candidate_generation.feature_space.explorekit_transformations import get_transformation_for_feature_space
import warnings
from fastsklearnfeature.instance_selection.instance_selection_cnn import sample_data_by_cnn
import copy
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import tqdm
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from functools import partial
import itertools
from sklearn.model_selection import KFold

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))



def grid(grid_search_parameters, preprocessed_folds, train_data, current_target,test_data, test_target, pipeline, classifier, score):
    hyperparam_to_score_list = {}

    my_keys = list(grid_search_parameters.keys())

    for parameter_combination in itertools.product(*[grid_search_parameters[k] for k in my_keys]):
        parameter_set = hashabledict(zip(my_keys, parameter_combination))
        hyperparam_to_score_list[parameter_set] = []
        for train_id, test_id in preprocessed_folds:
            clf = classifier(**parameter_set)
            my_train = pipeline.fit_transform(train_data[train_id], current_target[train_id])
            clf.fit(my_train, current_target[train_id])
            y_pred = clf.predict(pipeline.transform(train_data[test_id]))

            hyperparam_to_score_list[parameter_set].append(
                score._sign * score._score_func(current_target[test_id], y_pred, **score._kwargs))


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
        my_train = pipeline.fit_transform(train_data, current_target)
        clf.fit(my_train, current_target)
        y_pred = clf.predict(pipeline.transform(test_data))
        test_score = score._sign * score._score_func(test_target, y_pred, **score._kwargs)

        # np.save('/tmp/true_predictions', self.test_target)

    return best_mean_cross_val_score, test_score, best_param, y_pred

'''
def evaluate(candidate: CandidateFeature, classifier, grid_search_parameters, preprocessed_folds, score, train_data, current_target, train_X_all, train_y_all_target, test_data, test_target):
    candidate.runtime_properties['score'], candidate.runtime_properties['test_score'], candidate.runtime_properties['hyperparameters'], y_pred = grid(grid_search_parameters, preprocessed_folds, train_data, current_target, test_data, test_target, candidate.pipeline, classifier,
         score)


    return candidate
'''

def evaluate(candidate: CandidateFeature, classifier, grid_search_parameters, preprocessed_folds, score, train_data, current_target, train_X_all, train_y_all_target, test_data, test_target, cv_jobs=1):
    pipeline = Pipeline([('features', FeatureUnion(
        [
            (candidate.get_name(), candidate.pipeline)
        ])),
                         ('classifier', classifier())
                         ])

    refit = False
    if Config.get_default('score.test', 'False') == 'True' and not Config.get_default('instance.selection',
                                                                                      'False') == 'True':
        refit = True

    print(grid_search_parameters)
    clf = GridSearchCV(pipeline, grid_search_parameters, cv=preprocessed_folds, scoring=score, iid=False,
                       error_score='raise', refit=refit, n_jobs=cv_jobs)
    clf.fit(train_data, current_target) #dataset.splitted_values['train']
    candidate.runtime_properties['score'] = clf.best_score_
    candidate.runtime_properties['hyperparameters'] = clf.best_params_

    #for
    test_fold_predictions = []
    for fold in range(len(preprocessed_folds)):
        test_fold_predictions.append(clf.predict(train_data[preprocessed_folds[fold][1]]) == current_target[preprocessed_folds[fold][1]])
    candidate.runtime_properties['test_fold_predictions'] = test_fold_predictions

    if Config.get_default('score.test', 'False') == 'True' and len(test_data) > 0:
        if Config.get_default('instance.selection', 'False') == 'True':
            clf = GridSearchCV(pipeline, grid_search_parameters, cv=preprocessed_folds, scoring=score,
                               iid=False, error_score='raise', refit=True)

            clf.fit(train_X_all, train_y_all_target)
        candidate.runtime_properties['test_score'] = clf.score(test_data, test_target) #self.dataset.splitted_values['test']
    else:
        candidate.runtime_properties['test_score'] = 0.0

    return candidate

def evaluate_randomcv(candidate: CandidateFeature, classifier, grid_search_parameters, preprocessed_folds, score, train_data, current_target, train_X_all, train_y_all_target, test_data, test_target, cv_jobs=1):
    pipeline = Pipeline([('features', FeatureUnion(
        [
            (candidate.get_name(), candidate.pipeline)
        ])),
                         ('classifier', classifier())
                         ])

    refit = False
    if Config.get_default('score.test', 'False') == 'True' and not Config.get_default('instance.selection',
                                                                                      'False') == 'True':
        refit = True

    clf = RandomizedSearchCV(pipeline, grid_search_parameters, cv=preprocessed_folds, scoring=score, iid=False,
                       error_score='raise', refit=refit, n_jobs=cv_jobs, n_iter=100)
    clf.fit(train_data, current_target) #dataset.splitted_values['train']
    candidate.runtime_properties['score'] = clf.best_score_
    candidate.runtime_properties['hyperparameters'] = clf.best_params_

    if Config.get_default('score.test', 'False') == 'True':
        if Config.get_default('instance.selection', 'False') == 'True':
            clf = GridSearchCV(pipeline, grid_search_parameters, cv=preprocessed_folds, scoring=score,
                               iid=False, error_score='raise', refit=True)

            clf.fit(train_X_all, train_y_all_target)
        candidate.runtime_properties['test_score'] = clf.score(test_data, test_target) #self.dataset.splitted_values['test']
    else:
        candidate.runtime_properties['test_score'] = 0.0

    return candidate






class EvaluationFramework:
    def __init__(self, dataset_config, classifier=LogisticRegression, grid_search_parameters={'classifier__penalty': ['l2'],
                                                                                                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'classifier__solver': ['lbfgs'],
                                                                                                'classifier__class_weight': ['balanced'],
                                                                                                'classifier__max_iter': [10000],
                                                                                                'classifier__multi_class':['auto']
                                                                                                },
                 transformation_producer=get_transformation_for_feature_space
                 ):
        self.dataset_config = dataset_config
        self.classifier = classifier
        self.grid_search_parameters = grid_search_parameters
        self.transformation_producer = transformation_producer

    #generate all possible combinations of features
    def generate(self, seed=42):
        if type(self.reader) == type(None):
            s = None
            if isinstance(self.classifier(), ClassifierMixin):
                s = Splitter(train_fraction=[0.6, 10000000], valid_fraction=0.0, test_fraction=0.4, seed=seed)
            elif isinstance(self.classifier(), RegressorMixin):
                s = RandomSplitter(train_fraction=[0.6, 10000000], valid_fraction=0.0, test_fraction=0.4, seed=seed)
            else:
                pass

            self.dataset = Reader(self.dataset_config[0], self.dataset_config[1], s)
        else:
            self.dataset = self.reader
        self.raw_features = self.dataset.read()

        print("training:" + str(len(self.dataset.splitted_target['train'])))
        print("test:" + str(len(self.dataset.splitted_target['test'])))

        if Config.get_default('instance.selection', 'False') == 'True':
            self.train_X_all = copy.deepcopy(self.dataset.splitted_values['train'])
            self.train_y_all = copy.deepcopy(self.dataset.splitted_target['train'])

            self.dataset.splitted_values['train'], self.dataset.splitted_target['train'] = sample_data_by_cnn(self.dataset.splitted_values['train'], self.dataset.splitted_target['train'])
            print("training:" + str(len(self.dataset.splitted_target['train'])))
        else:
            self.train_X_all = self.dataset.splitted_values['train']
            self.train_y_all = self.dataset.splitted_target['train']


    #rank and select features
    def random_select(self, k: int):
        arr = np.arange(len(self.candidates))
        np.random.shuffle(arr)
        return arr[0:k]

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
            for train, test in StratifiedKFold(n_splits=self.folds, random_state=42).split(
                    self.dataset.splitted_values['train'],
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


    '''
    def evaluate_candidates(self, candidates: List[CandidateFeature]) -> List[CandidateFeature]:
        pool = mp.Pool(processes=int(Config.get_default("parallelism", mp.cpu_count())))



        my_function = partial(evaluate, classifier=self.classifier,
                              grid_search_parameters=self.grid_search_parameters,
                              preprocessed_folds=self.preprocessed_folds,
                              score=self.score,
                              train_data=self.dataset.splitted_values['train'],
                              current_target=self.current_target,
                              train_X_all=self.train_X_all,
                              train_y_all_target=self.train_y_all_target,
                              test_data=self.dataset.splitted_values['test'],
                              test_target=self.test_target)

        if Config.get_default("show_progess", 'True') == 'True':
            results = []
            for x in tqdm.tqdm(pool.imap_unordered(my_function, candidates), total=len(candidates)):
                results.append(x)
        else:
            results = pool.map(my_function, candidates)


        return results
    '''

    def evaluate_candidates(self, candidates: List[CandidateFeature], my_folds) -> List[CandidateFeature]:
        my_function = partial(evaluate, classifier=self.classifier,
                              grid_search_parameters=self.grid_search_parameters,
                              preprocessed_folds=my_folds,
                              score=self.score,
                              train_data=self.dataset.splitted_values['train'],
                              current_target=self.train_y_all_target,
                              train_X_all=self.train_X_all,
                              train_y_all_target=self.train_y_all_target,
                              test_data=self.dataset.splitted_values['test'],
                              test_target=self.test_target)

        results = []
        for can in candidates:
            results.append(my_function(can))
        return results

    def evaluate_candidates_detail(self, candidates: List[CandidateFeature], my_folds, cv_jobs) -> List[CandidateFeature]:
        my_function = partial(evaluate, classifier=self.classifier,
                              grid_search_parameters=self.grid_search_parameters,
                              preprocessed_folds=my_folds,
                              score=self.score,
                              train_data=self.dataset.splitted_values['train'],
                              current_target=self.train_y_all_target,
                              train_X_all=self.train_X_all,
                              train_y_all_target=self.train_y_all_target,
                              test_data=self.dataset.splitted_values['test'],
                              test_target=self.test_target,
                              cv_jobs=cv_jobs)

        results = []
        for can in candidates:
            results.append(my_function(can))
        return results

    def evaluate_candidates_randomcv(self, candidates: List[CandidateFeature], my_folds, cv_jobs) -> List[CandidateFeature]:
        my_function = partial(evaluate_randomcv, classifier=self.classifier,
                              grid_search_parameters=self.grid_search_parameters,
                              preprocessed_folds=my_folds,
                              score=self.score,
                              train_data=self.dataset.splitted_values['train'],
                              current_target=self.train_y_all_target,
                              train_X_all=self.train_X_all,
                              train_y_all_target=self.train_y_all_target,
                              test_data=self.dataset.splitted_values['test'],
                              test_target=self.test_target,
                              cv_jobs=cv_jobs)

        results = []
        for can in candidates:
            results.append(my_function(can))
        return results


    '''
    def evaluate_candidates(self, candidates):
        results = []
        for c in candidates:
            results.append(self.evaluate_single_candidate(c))
        return results

    '''






