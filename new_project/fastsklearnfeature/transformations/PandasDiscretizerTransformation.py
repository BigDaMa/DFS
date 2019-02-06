from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PandasDiscretizerTransformation(BaseEstimator, TransformerMixin, NumericUnaryTransformation):
    def __init__(self, number_bins, qbucket=False):
        self.number_bins = number_bins
        self.qbucket = qbucket
        name = 'Discretizer'
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, X, y=None):
        if not self.qbucket:
            _, self.bins = pd.cut(X[:,0], bins=self.number_bins, retbins=True, labels=range(self.number_bins))
        else:
            _, self.bins = pd.qcut(X[:,0], q=self.number_bins, retbins=True, labels=range(self.number_bins))
        return self

    def transform(self, X):
        bucket_labels = pd.cut(X[:,0], bins=self.bins, labels=range(self.number_bins),
                               include_lowest=True).__array__()
        bucket_labels[np.isnan(bucket_labels)] = -1
        return np.reshape(bucket_labels, (len(X), 1))