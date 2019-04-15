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


def evaluate(candidate: CandidateFeature, classifier, grid_search_parameters, preprocessed_folds, score, train_data, current_target, train_X_all, train_y_all_target, test_data, test_target):
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

    clf = GridSearchCV(pipeline, grid_search_parameters, cv=preprocessed_folds, scoring=score, iid=False,
                       error_score='raise', refit=refit)
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
    def generate(self):
        s = None
        if isinstance(self.classifier(), ClassifierMixin):
            s = Splitter(train_fraction=[0.6, 10000000], valid_fraction=0.0, test_fraction=0.4, seed=42)
        elif isinstance(self.classifier(), RegressorMixin):
            s = RandomSplitter(train_fraction=[0.6, 10000000], valid_fraction=0.0, test_fraction=0.4, seed=42)
        else:
            pass

        self.dataset = Reader(self.dataset_config[0], self.dataset_config[1], s)
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

    def generate_target(self, folds=10):
        current_target = self.dataset.splitted_target['train']

        label_encoder = LabelEncoder()
        label_encoder.fit(current_target)

        self.current_target = label_encoder.transform(current_target)

        if Config.get_default('score.test', 'False') == 'True':
            self.test_target = label_encoder.transform(self.dataset.splitted_target['test'])
            self.train_y_all_target = label_encoder.transform(self.train_y_all)


        self.preprocessed_folds = []
        for train, test in StratifiedKFold(n_splits=folds, random_state=42).split(self.dataset.splitted_values['train'],
                                                                               self.current_target):
            self.preprocessed_folds.append((train, test))


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
    def evaluate_candidates(self, candidates):
        results = []
        for c in candidates:
            results.append(self.evaluate_single_candidate(c))
        return results

    '''






