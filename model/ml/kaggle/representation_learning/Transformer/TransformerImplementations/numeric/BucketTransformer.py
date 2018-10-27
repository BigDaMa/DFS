import numpy as np
import pandas as pd
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class BucketTransformer(NumericTransformer):

    def __init__(self, column_id, number_bins=10, qbucket=False):
        name = ""
        if qbucket:
            name += 'q'
        name += "bucket"

        NumericTransformer.__init__(self, column_id, name)
        self.number_bins = number_bins
        self.qbucket = qbucket

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1

        if not self.qbucket:
            _, self.bins = pd.cut(column_data, bins=self.number_bins, retbins=True, labels=range(self.number_bins))
        else:
            _, self.bins = pd.qcut(column_data, q=self.number_bins, retbins=True, labels=range(self.number_bins))


    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        bucket_labels = pd.cut(column_data, bins=self.bins, labels=range(self.number_bins), include_lowest=True).__array__()

        return np.matrix(bucket_labels).T

    def __str__(self):
        return self.__class__.__name__ + "_dimensionality_" + str(self.number_bins) + "_qbuckets_" + str(self.qbucket)