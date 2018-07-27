import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class IdentityTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "identity")

    def transform(self, dataset, ids):
        matrix = np.matrix(dataset[dataset.columns[self.column_id]].values).T[ids, :]
        return matrix