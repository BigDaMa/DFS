import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer
from sklearn.preprocessing import add_dummy_feature

class DummyTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "dummy")


    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        return add_dummy_feature(np.matrix(column_data).T)

    def get_feature_names(self, dataset):
        return [str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + 'dummy', str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + 'identity']