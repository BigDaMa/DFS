from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

from fastsklearnfeature.feature_selection.ComplexityDrivenFeatureConstruction import ComplexityDrivenFeatureConstruction
from fastsklearnfeature.reader.ScikitReader import ScikitReader
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline as ImbalancePipeline
from imblearn.over_sampling import SMOTE
from fastsklearnfeature.transformations.sampling.PipelineTransformation import PipelineTransformation

import warnings


class ComplexityDrivenFeatureConstructionScikit:

    def __init__(self, max_time_secs=None, scoring=make_scorer(f1_score, average='micro'), model=LogisticRegression, parameter_grid={'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class':['auto']}, n_jobs=None, epsilon=-np.inf, feature_names = None, feature_is_categorical=None):
        self.fe = None
        self.max_feature_rep: CandidateFeature = None
        self.pipeline = None
        self.max_time_secs = max_time_secs
        self.scoring = scoring
        self.model = model
        self.parameter_grid = parameter_grid
        self.n_jobs = n_jobs
        self.epsilon = epsilon
        self.feature_names = feature_names
        self.feature_is_categorical = feature_is_categorical

    def fit(self, features, target, sample_weight=None, groups=None):
        #self.fe = ComplexityDrivenFeatureConstruction(None, reader=ScikitReader(features, target, feature_names=self.feature_names, feature_is_categorical=self.feature_is_categorical),
        #                                              score=self.scoring, c_max=np.inf, folds=10, max_seconds=self.max_time_secs, classifier=self.model, grid_search_parameters=self.parameter_grid, n_jobs=self.n_jobs, epsilon=self.epsilon)

        self.fe = ComplexityDrivenFeatureConstruction(None, reader=ScikitReader(features, target,
                                                                                feature_names=self.feature_names,
                                                                                feature_is_categorical=self.feature_is_categorical),
                                                      score=self.scoring, c_max=6, folds=10,
                                                      max_seconds=self.max_time_secs, classifier=self.model,
                                                      grid_search_parameters=self.parameter_grid, n_jobs=self.n_jobs,
                                                      epsilon=0.0)

        self.max_feature_rep = self.fe.run()

        self.pipeline = self.generate_pipeline().fit(features, target)



    def generate_pipeline(self):
        best_hyperparameters = self.max_feature_rep.runtime_properties['hyperparameters']

        all_keys = list(best_hyperparameters.keys())
        for k in all_keys:
            if 'classifier__' in k:
                best_hyperparameters[k[12:]] = best_hyperparameters.pop(k)

        '''
        my_pipeline = ImbalancePipeline([('f', PipelineTransformation(self.max_feature_rep.pipeline)),
                                ('smote', SMOTE()),
                                ('c', self.fe.classifier(**best_hyperparameters))
                                ])
        '''
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










