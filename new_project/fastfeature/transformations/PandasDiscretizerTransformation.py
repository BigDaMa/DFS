from fastfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
import pandas as pd
import numpy as np

class PandasDiscretizerTransformation(NumericUnaryTransformation):
    def __init__(self, number_bins, qbucket=False):
        self.number_bins = number_bins
        self.qbucket = qbucket
        name = 'Discretizer'
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, data):
        if not self.qbucket:
            _, self.bins = pd.cut(np.matrix(data).A1, bins=self.number_bins, retbins=True, labels=range(self.number_bins))
        else:
            _, self.bins = pd.qcut(data, q=self.number_bins, retbins=True, labels=range(self.number_bins))

    def transform(self, data):
        bucket_labels = pd.cut(np.matrix(data).A1, bins=self.bins, labels=range(self.number_bins),
                               include_lowest=True).__array__()
        return bucket_labels
