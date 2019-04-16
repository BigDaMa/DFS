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
import fastsklearnfeature.feature_selection.evaluation.my_globale_module as my_globale_module
import os
import psutil
import sys




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





















