from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import time
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.linear_model import LogisticRegression
import pickle
import multiprocessing as mp
from fastsklearnfeature.configuration.Config import Config
import itertools
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
import copy
from fastsklearnfeature.candidate_generation.feature_space.division import get_transformation_for_division
from fastsklearnfeature.feature_selection.evaluation.CachedEvaluationFramework import CachedEvaluationFramework
import sympy
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import fastsklearnfeature.feature_selection.evaluation.my_globale_module as my_globale_module
from fastsklearnfeature.feature_selection.evaluation.run_evaluation import evaluate_candidates
from fastsklearnfeature.feature_selection.openml_wrapper.pipeline2openml import candidate2openml

from fastsklearnfeature.feature_selection.ComplexityDrivenFeatureConstruction import ComplexityDrivenFeatureConstruction
from fastsklearnfeature.reader.ScikitReader import ScikitReader
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import accuracy_score


import warnings


class ComplexityDrivenFeatureConstructionScikit:

    def __init__(self, max_time_secs=None, scoring=make_scorer(f1_score, average='micro'), model=LogisticRegression, parameter_grid={'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class':['auto']}, n_jobs=None):
        self.fe = None
        self.max_feature_rep: CandidateFeature = None
        self.pipeline = None
        self.max_time_secs = max_time_secs
        self.scoring = scoring
        self.model = model
        self.parameter_grid = parameter_grid
        self.n_jobs = n_jobs

    def fit(self, features, target, sample_weight=None, groups=None):
        self.fe = ComplexityDrivenFeatureConstruction(None, reader=ScikitReader(features, target),
                                                      score=self.scoring, c_max=np.inf, folds=10, max_seconds=self.max_time_secs, classifier=self.model, grid_search_parameters=self.parameter_grid, n_jobs=self.n_jobs)

        self.max_feature_rep = self.fe.run()

        self.pipeline = self.generate_pipeline().fit(features, target)



    def generate_pipeline(self):
        best_hyperparameters = self.max_feature_rep.runtime_properties['hyperparameters']

        all_keys = list(best_hyperparameters.keys())
        for k in all_keys:
            if 'classifier__' in k:
                best_hyperparameters[k[12:]] = best_hyperparameters.pop(k)

        my_pipeline = Pipeline([('f', self.max_feature_rep.pipeline),
                                ('c', self.fe.classifier(**best_hyperparameters))
                                ])
        return my_pipeline


    def predict(self, features):
        return self.pipeline.predict(features)

    def predict_proba(self, features):
        return self.pipeline.predict_proba(features)



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    fe = ComplexityDrivenFeatureConstructionScikit()
    fe.fit(X_train, y_train)

    y_pred = fe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print('Accuracy: ' + str(acc))










