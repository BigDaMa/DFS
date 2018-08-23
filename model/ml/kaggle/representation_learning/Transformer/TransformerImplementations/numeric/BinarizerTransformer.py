import numpy as np
from sklearn.preprocessing import Binarizer
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class BinarizerTransformer(NumericTransformer):

    def __init__(self, column_id, threshold=0.0):
        NumericTransformer.__init__(self, column_id, "binary")
        self.threshold = threshold
        self.model = Binarizer(self.threshold)

    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        return np.matrix(self.model.transform(column_data.reshape(1, -1))).T
