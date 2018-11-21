from fastfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from fastfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
from sklearn.preprocessing import MinMaxScaler

class MinMaxScalingTransformation(NumericUnaryTransformation):
    def __init__(self):
        name = 'MinMaxScaling'
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, data):
        self.minmaxscaler = MinMaxScaler()
        self.minmaxscaler.fit(data.reshape(-1, 1))

    def transform(self, data):
        return self.minmaxscaler.transform(data.reshape(-1, 1))
