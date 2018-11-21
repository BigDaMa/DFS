import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer
from scipy import stats
import math



class SigmoidTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "sigmoid", 1)

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        self.sigmoid = np.vectorize(sigmoid)


    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        return np.matrix(self.sigmoid(column_data)).T