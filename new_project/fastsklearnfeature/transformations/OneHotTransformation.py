from fastsklearnfeature.transformations.Transformation import Transformation
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
import sympy

class equals_string(sympy.Function):
    @classmethod
    def eval(cls, value, string_value):
        if isinstance(value, equals_string):
            return 0

class OneHotTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, value: str, value_id: str, raw_feature: RawFeature):
        self.value = value
        self.value_id = value_id
        self.is_contained = False
        self.raw_feature = raw_feature
        Transformation.__init__(self, 'equals_string',
                 number_parent_features=1,
                 output_dimensions=1,
                 parent_feature_order_matters=True,
                 parent_feature_repetition_is_allowed=False)

    #0th feature will be aggregated, 1th-nth = key attributes
    def fit(self, X, y=None):
        if self.value in X:
            self.is_contained = True
        return self

    def transform(self, X):
        if self.is_contained:
            return X == self.value
        else:
            return np.zeros(X.shape)

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if feature_combination[0] == self.raw_feature:
            return True
        return False

    def get_name(self, candidate_feature_names):
        return "(" + str(self.raw_feature) + ' == "' + str(self.value) + '")'

    def get_sympy_representation(self, input_attributes):
        return equals_string(input_attributes[0], sympy.Symbol('X' + str(self.raw_feature.column_id) + 'S' + str(self.value_id)))

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = np.bool

        try:
            # missing values properties
            properties['missing_values'] = False

            # range properties
            properties['has_zero'] = True
            properties['min'] = 0.0
            properties['max'] = 1.0
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = 2
        return properties



