from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from autofeat import AutoFeatRegression
import pandas as pd

class SissoTransformer(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features, feature_names, transformations):
        Transformation.__init__(self, 'selectk',
                 number_parent_features, output_dimensions=None,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)
        self.feature_names = feature_names
        self.transformations = transformations
        self.model = AutoFeatRegression(feateng_cols=feature_names, transformations=self.transformations)

    def fit(self, X, y=None):
        self.model.fit(pd.DataFrame(data=X, columns=self.feature_names), y)
        return self

    def transform(self, data):
        return self.model.transform(pd.DataFrame(data=data, columns=self.feature_names))


