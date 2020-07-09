from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

from fastsklearnfeature.feature_selection.ComplexityDrivenFeatureConstruction import ComplexityDrivenFeatureConstruction
from fastsklearnfeature.reader.ScikitReader import ScikitReader
from sklearn.base import BaseEstimator, TransformerMixin

from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from fastsklearnfeature.transformations.MinusTransformation import MinusTransformation
from fastsklearnfeature.candidate_generation.feature_space.division import get_transformation_for_division
from fastsklearnfeature.transformations.MyImputationTransformation import ImputationTransformation
from fastsklearnfeature.transformations.HigherOrderCommutativeTransformation import HigherOrderCommutativeTransformation
import pickle
import numpy as np
import sympy
from sympy import S

class ConstructionTransformer(TransformerMixin, BaseEstimator):

    def __init__(self,
                 c_max=2,
                 max_time_secs=None,
                 scoring=make_scorer(f1_score, average='micro'),
                 model=None,
                 parameter_grid={'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class':['auto']},
                 n_jobs=None,
                 epsilon=0.0,
                 feature_names=None,
                 feature_is_categorical=None,
                 cv=5,
                 transformation_producer=get_transformation_for_division):
        self.c_max = c_max
        self.max_time_secs = max_time_secs
        self.scoring = scoring
        self.model = model
        self.parameter_grid = parameter_grid
        self.n_jobs = n_jobs
        self.epsilon = epsilon
        self.feature_names = feature_names
        self.feature_is_categorical = feature_is_categorical
        self.cv = cv
        self.transformation_producer = transformation_producer

    def fit(self, X, y=None):
        fe = ComplexityDrivenFeatureConstruction(None, reader=ScikitReader(X, y,
                                                                                feature_names=self.feature_names,
                                                                                feature_is_categorical=self.feature_is_categorical),
                                                      score=self.scoring, c_max=self.c_max, folds=self.cv,
                                                      max_seconds=self.max_time_secs, classifier=self.model.__class__,
                                                      grid_search_parameters=self.parameter_grid, n_jobs=self.n_jobs,
                                                      epsilon=self.epsilon, remove_parents=False,transformation_producer=self.transformation_producer)

        fe.run()

        numeric_representations = []
        for r in fe.all_representations:
            if 'score' in r.runtime_properties:
                if not 'object' in str(r.properties['type']):
                    if not isinstance(r.transformation, MinMaxScalingTransformation):
                        #if not (isinstance(r.transformation, HigherOrderCommutativeTransformation) and r.transformation.method == np.nansum):
                        if isinstance(r.sympy_representation, sympy.Mul):
                            found = False
                            for e in r.sympy_representation._args:
                                if e == S.NegativeOne:
                                    found = True
                            if found == False:
                                numeric_representations.append(r)
                        else:
                            numeric_representations.append(r)

        self.numeric_features = numeric_representations


        my_list = []
        for ff in self.numeric_features:
            my_list.append(str(ff))

        with open('/tmp/names.pickle', 'wb') as f:
            pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)


        all_features = CandidateFeature(IdentityTransformation(-1), numeric_representations)


        #all_imputation = CandidateFeature(ImputationTransformation(), [all_features])
        all_standardized = CandidateFeature(MinMaxScalingTransformation(), [all_features])


        #all_standardized = CandidateFeature(MinMaxScalingTransformation(), [all_features])

        self.pipeline_ = all_standardized.pipeline

        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline_.transform(X)


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)






