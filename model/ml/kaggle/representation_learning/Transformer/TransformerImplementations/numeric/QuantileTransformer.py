import numpy as np
from sklearn import preprocessing
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class QuantileTransformer(NumericTransformer):

    def __init__(self, column_id, output_distribution='normal'):
        NumericTransformer.__init__(self, column_id, "quantile")
        self.seed = 42
        self.output_distribution = output_distribution #'uniform'

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        newy = column_data.reshape(-1, 1)
        self.quantile_transformer = preprocessing.QuantileTransformer(random_state=self.seed, output_distribution=self.output_distribution).fit(newy)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        newy = column_data.reshape(-1, 1)
        return self.quantile_transformer.transform(newy)
