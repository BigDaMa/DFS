from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sympy
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List

class HigherOrderCommutativeTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, method, sympy_method, number_parent_features):
        self.method = method
        self.sympy_method = sympy_method
        Transformation.__init__(self, self.method.__name__,
                 number_parent_features, output_dimensions=1,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=True)


    def transform(self, X):
        try:
            return np.reshape(self.method(X, axis=1), (len(X), 1))
        except Exception as e:
            print('HigherOrderCommutativeTransformation' + str(e))


    def is_applicable(self, feature_combination):
        #the aggregated column has to be numeric
        for element in feature_combination:
            if not ('float' in str(element.properties['type']) \
                or 'int' in str(element.properties['type']) \
                or 'bool' in str(element.properties['type'])):
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
            properties['missing_values'] = False

            if self.method == np.nanprod:
                properties['has_zero'] = any([p.properties['has_zero'] for p in parents])
            else:
                properties['has_zero'] = 0 in training_data

            properties['min'] = np.nanmin(training_data)
            properties['max'] = np.nanmax(training_data)
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = len(np.unique(training_data))
        return properties