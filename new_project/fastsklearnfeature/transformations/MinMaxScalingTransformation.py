from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class MinMaxScalingTransformation(BaseEstimator, TransformerMixin, NumericUnaryTransformation):
    def __init__(self):
        name = 'MinMaxScaling'
        self.minmaxscaler = MinMaxScaler()
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, X, y=None):
        self.minmaxscaler.fit(X)
        return self

    def transform(self, X):
        return self.minmaxscaler.transform(X)
