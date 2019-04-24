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

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(MinusTransformation, self).is_applicable(feature_combination):
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
            properties['missing_values'] = False

            properties['has_zero'] = any([p.properties['has_zero'] for p in parents])

            properties['min'] = np.nanmin(training_data)
            properties['max'] = np.nanmax(training_data)
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = len(np.unique(training_data))
        return properties