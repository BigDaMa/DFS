import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class LogTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "log", 1)


    def transform1(self, column_data):
        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = 0.000001

        column_data[column_data <= 0] = 0.000001

        return np.matrix(np.log(column_data)).T

    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)

        return self.transform1(column_data)