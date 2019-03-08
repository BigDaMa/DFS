from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE

class FeatureEliminationTransformer(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features,  output_dimensions, estimator):
        Transformation.__init__(self, 'selectk',
                 number_parent_features, output_dimensions=output_dimensions,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)
        self.model = RFE(estimator, n_features_to_select=output_dimensions)

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, data):
        return self.model.transform(data)


