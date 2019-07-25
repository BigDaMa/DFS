from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
import sympy

class LogTransformation(BaseEstimator, TransformerMixin, NumericUnaryTransformation):
    def __init__(self):
        name = 'log'
        NumericUnaryTransformation.__init__(self, name)

    def transform(self, X):
        return np.log(X)

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(LogTransformation, self).is_applicable(feature_combination):
            return False
        if feature_combination[0].properties['has_zero']:
            return False
        if feature_combination[0].properties['min'] <= 0:
            return False
        if 'missing_values' in feature_combination[0].properties and feature_combination[0].properties['missing_values']:
            return False

        return True

    def get_sympy_representation(self, input_attributes):
        return sympy.functions.log(input_attributes[0])

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            # missing values properties
            properties['missing_values'] = parents[0].properties['missing_values']

            properties['has_zero'] = False

            properties['min'] = np.log(parents[0].properties['min'])
            properties['max'] = np.log(parents[0].properties['max'])
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = parents[0].properties['number_distinct_values']
        return properties