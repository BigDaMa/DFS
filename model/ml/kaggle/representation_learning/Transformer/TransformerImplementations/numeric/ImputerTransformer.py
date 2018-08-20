import numpy as np
from sklearn.preprocessing import Imputer
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class ImputerTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "imputer")
        self.seed = 42
        self.model = Imputer()

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).T
        self.model.fit(column_data)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).T
        return np.matrix(self.model.transform(column_data))
