from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
import featuretools as ft

class DFSTransformer(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features, output_dimensions):
        Transformation.__init__(self, 'DFS',
                 number_parent_features, output_dimensions=output_dimensions,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)

        self.model = ft.wrappers.DFSTransformer(entityset=es, target_entity="customers", max_features=output_dimensions)

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, data):
        return self.model.transform(data)


