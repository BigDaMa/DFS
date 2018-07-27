import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class LogTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "log")


    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        return np.log(column_data)