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
























