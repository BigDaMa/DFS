from fastsklearnfeature.transformations.BinaryTransformation import BinaryTransformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class NonCommutativeBinaryTransformation(BaseEstimator, TransformerMixin, BinaryTransformation):
    def __init__(self, method):
        self.method = method
        BinaryTransformation.__init__(self, self.method.__name__, output_dimensions=1,
                 parent_feature_order_matters=True, parent_feature_repetition_is_allowed=False)

    def transform(self, X):
        return np.reshape(self.method(X[:,0], X[:,1]), (len(X), 1))