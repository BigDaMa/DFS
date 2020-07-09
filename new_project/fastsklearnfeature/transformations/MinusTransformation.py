from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
import sympy

class MinusTransformation(BaseEstimator, TransformerMixin, NumericUnaryTransformation):
    def __init__(self):
        name = '-1*'
        NumericUnaryTransformation.__init__(self, name)

    def transform(self, X):
        return -1 * X

    def fit(self, X, y=None):
        return self

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(MinusTransformation, self).is_applicable(feature_combination):
            return False
        if 'missing_values' in feature_combination[0].properties and feature_combination[0].properties['missing_values']:
            return False

        return True

    def get_sympy_representation(self, input_attributes):
        return sympy.Mul(input_attributes[0], -1)

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            # missing values properties
            properties['missing_values'] = parents[0].properties['missing_values']

            properties['has_zero'] = parents[0].properties['has_zero']

            properties['min'] = -1 * parents[0].properties['max']
            properties['max'] = -1 * parents[0].properties['min']
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = parents[0].properties['number_distinct_values']
        return properties