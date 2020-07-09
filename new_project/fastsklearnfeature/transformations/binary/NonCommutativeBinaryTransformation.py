from fastsklearnfeature.transformations.BinaryTransformation import BinaryTransformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sympy
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List

class NonCommutativeBinaryTransformation(BaseEstimator, TransformerMixin, BinaryTransformation):
    def __init__(self, method, sympy_method):
        self.method = method
        self.sympy_method = sympy_method
        BinaryTransformation.__init__(self, self.method.__name__, output_dimensions=1,
                 parent_feature_order_matters=True, parent_feature_repetition_is_allowed=False)

    def transform(self, X):
        return np.reshape(self.method(X[:,0], X[:,1]), (len(X), 1))

    def fit(self, X, y=None):
        return self

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        #the aggregated column has to be numeric
        for i in range(len(feature_combination)):
            if not ('float' in str(feature_combination[i].properties['type']) \
                or 'int' in str(feature_combination[i].properties['type']) \
                or 'bool' in str(feature_combination[i].properties['type'])):
                return False

        if self.method == np.divide and 'has_zero' in feature_combination[1].properties and feature_combination[1].properties['has_zero']:
            return False

        return True

    def get_sympy_representation(self, input_attributes):
        return sympy.factor(self.sympy_method(*input_attributes))

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            # missing values properties
            #properties['missing_values'] = any([parents[0].properties['missing_values'],parents[1].properties['missing_values']])

            if self.method == np.divide:
                properties['has_zero'] = False
            else:
                properties['has_zero'] = 0 in training_data

            properties['min'] = np.nanmin(training_data)
            properties['max'] = np.nanmax(training_data)
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = len(np.unique(training_data))
        return properties