import numpy as np
from sklearn.preprocessing import StandardScaler
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class ScaleTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "scale")

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        #bad_indices = np.where(np.isinf(np.array(column_data)))
        #column_data[bad_indices] = -1
        #column_data = np.nan_to_num(column_data)
        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        newy = column_data.reshape(-1, 1)



        self.scaler = StandardScaler()
        self.scaler.fit(newy)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        newy = column_data.reshape(-1, 1)
        return self.scaler.transform(newy)