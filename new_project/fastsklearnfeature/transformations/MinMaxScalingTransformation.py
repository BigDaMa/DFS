from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List

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

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(MinMaxScalingTransformation, self).is_applicable(feature_combination):
            return False
        if isinstance(feature_combination[0].transformation, MinMaxScalingTransformation):
            return False

        return True
