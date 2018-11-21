from fastfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from fastfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
from sklearn.preprocessing._discretization import KBinsDiscretizer

class DiscretizerTransformation(NumericUnaryTransformation):
    def __init__(self, number_bins, strategy='uniform'):
        self.number_bins = number_bins
        self.strategy = strategy
        name = 'Discretizer'
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, data):
        self.discretizer = KBinsDiscretizer(self.number_bins, encode='ordinal', strategy=self.strategy)
        self.discretizer.fit(data.reshape(-1, 1))

    def transform(self, data):
        return self.discretizer.transform(data.reshape(-1, 1))
