from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

class SelectKBestTransformer(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features,  output_dimensions):
        Transformation.__init__(self, 'selectk',
                 number_parent_features, output_dimensions=output_dimensions,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)
        self.model = SelectKBest(mutual_info_classif, k=output_dimensions)

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, data):
        return self.model.transform(data)


