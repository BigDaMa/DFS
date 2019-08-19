from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
import sympy
import numpy

class standardscale(sympy.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, standardscale): #idempotent
            return x

class StandardScalingTransformation(BaseEstimator, TransformerMixin, NumericUnaryTransformation):
    def __init__(self):
        name = 'StandardScaling'
        self.standardscaler = StandardScaler()
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, X, y=None):
        self.standardscaler.fit(X)
        return self

    def transform(self, X):
        return self.standardscaler.transform(X)

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        return True

    def get_sympy_representation(self, input_attributes):
        return standardscale(input_attributes[0])

    #TODO: fix this
    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            properties['missing_values'] = parents[0].properties['missing_values']

            # range properties
            properties['has_zero'] = True
            properties['min'] = 0.0
            properties['max'] = 1.0
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = parents[0].properties['number_distinct_values']
        return properties