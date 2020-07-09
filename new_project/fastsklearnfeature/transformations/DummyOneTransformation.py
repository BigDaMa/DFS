from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DummyOneTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features):
        Transformation.__init__(self, 'identity',
                 number_parent_features, output_dimensions=number_parent_features,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)

    def transform(self, X):
        return np.ones((len(X), 1))

    def fit(self, X, y=None):
        return self
