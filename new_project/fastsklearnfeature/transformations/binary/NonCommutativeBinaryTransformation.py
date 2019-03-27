from fastsklearnfeature.transformations.BinaryTransformation import BinaryTransformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class NonCommutativeBinaryTransformation(BaseEstimator, TransformerMixin, BinaryTransformation):
    def __init__(self, method, sympy_method):
        self.method = method
        self.sympy_method = sympy_method
        BinaryTransformation.__init__(self, self.method.__name__, output_dimensions=1,
                 parent_feature_order_matters=True, parent_feature_repetition_is_allowed=False)

    def transform(self, X):
        return np.reshape(self.method(X[:,0], X[:,1]), (len(X), 1))

    def is_applicable(self, feature_combination):
        #the aggregated column has to be numeric
        for i in range(len(feature_combination)):
            if not ('float' in str(feature_combination[i].properties['type']) \
                or 'int' in str(feature_combination[i].properties['type']) \
                or 'bool' in str(feature_combination[i].properties['type'])):
                return False
        return True

    def get_sympy_representation(self, input_attributes):
        return self.sympy_method(*input_attributes)