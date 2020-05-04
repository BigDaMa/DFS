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

class ConstructionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, c_max=2, max_time_secs=None, scoring=make_scorer(f1_score, average='micro'), model=None, parameter_grid={'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class':['auto']}, n_jobs=None, epsilon=0.0, feature_names=None, feature_is_categorical=None, cv=5):
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


    def fit(self, X, y=None):
        fe = ComplexityDrivenFeatureConstruction(None, reader=ScikitReader(X, y,
                                                                                feature_names=self.feature_names,
                                                                                feature_is_categorical=self.feature_is_categorical),
                                                      score=self.scoring, c_max=self.c_max, folds=self.cv,
                                                      max_seconds=self.max_time_secs, classifier=self.model.__class__,
                                                      grid_search_parameters=self.parameter_grid, n_jobs=self.n_jobs,
                                                      epsilon=self.epsilon, remove_parents=False)

        fe.run()

        numeric_representations = []
        for r in fe.all_representations:
            if 'score' in r.runtime_properties:
                if not 'object' in str(r.properties['type']):
                    if not isinstance(r.transformation, MinusTransformation):
                        numeric_representations.append(r)

        all_features = CandidateFeature(IdentityTransformation(-1), numeric_representations)
        all_standardized = CandidateFeature(MinMaxScalingTransformation(), [all_features])

        self.pipeline_ = all_standardized.pipeline

        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline_.transform(X)






