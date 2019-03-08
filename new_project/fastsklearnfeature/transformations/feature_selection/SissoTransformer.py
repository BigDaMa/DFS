from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from autofeat import AutoFeatRegression

class SissoTransformer(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features):
        Transformation.__init__(self, 'selectk',
                 number_parent_features, output_dimensions=None,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)
        self.model = AutoFeatRegression()

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, data):
        return self.model.transform(data)


