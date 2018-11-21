import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.binary.BinaryTransformer import BinaryTransformer


class Addition(BinaryTransformer):

    def __init__(self, column_a, column_b):
        BinaryTransformer.__init__(self, column_a, column_b, "addition", 1)


    def transform(self, dataset, ids):
        column_a_data = np.array(dataset.values[ids, self.column_a], dtype=np.float64)
        column_b_data = np.array(dataset.values[ids, self.column_b], dtype=np.float64)

        return self.transform1(column_a_data, column_b_data)

    def transform1(self, a, b):
        return a + b