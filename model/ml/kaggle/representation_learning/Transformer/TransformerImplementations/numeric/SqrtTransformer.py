import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class SqrtTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "sqrt", 1)

    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)

        res = np.sqrt(column_data)

        where_are_NaNs = np.isnan(res)
        res[where_are_NaNs] = -1

        return np.matrix(res).T