from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class HigherOrderCommutativeTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, method, number_parent_features):
        self.method = method
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
        for i in range(len(feature_combination)):
            if not ('float' in str(feature_combination[i].properties['type']) \
                or 'int' in str(feature_combination[i].properties['type']) \
                or 'bool' in str(feature_combination[i].properties['type'])):
                return False
        return True
