import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class CosTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "log")


    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        return np.matrix(np.cos(column_data)).T