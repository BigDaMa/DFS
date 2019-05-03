from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
import sympy
import numpy

class scale(sympy.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, scale): #idempotent
            return x

class MinMaxScalingTransformation(BaseEstimator, TransformerMixin, NumericUnaryTransformation):
    def __init__(self):
        name = 'MinMaxScaling'
        self.minmaxscaler = MinMaxScaler()
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, X, y=None):
        self.minmaxscaler.fit(X)
        return self

    def transform(self, X):
        return self.minmaxscaler.transform(X)

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(MinMaxScalingTransformation, self).is_applicable(feature_combination):
            return False
        if isinstance(feature_combination[0].transformation, MinMaxScalingTransformation):
            return False
        if 'min' in feature_combination[0].properties and feature_combination[0].properties['min'] == 0.0 and \
           'max' in feature_combination[0].properties and feature_combination[0].properties['max'] == 1.0:
            return False

        return True

    def get_sympy_representation(self, input_attributes):
        return scale(input_attributes[0])

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            # missing values properties
            if 'missing_values' in parents[0].properties:
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