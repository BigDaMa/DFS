from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from skrebate import MultiSURF
from skrebate import SURF
from skrebate import ReliefF

class skrebateTransformer(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features,  output_dimensions):
        Transformation.__init__(self, 'skrebate',
                 number_parent_features, output_dimensions=output_dimensions,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)
        #self.model = MultiSURF(n_features_to_select=output_dimensions)
        #self.model = SURF(n_features_to_select=output_dimensions)
        self.model = ReliefF(n_features_to_select=output_dimensions, n_neighbors=100)

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, data):
        return self.model.transform(data)


