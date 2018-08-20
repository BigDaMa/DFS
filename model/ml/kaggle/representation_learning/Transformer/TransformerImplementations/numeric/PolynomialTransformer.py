import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class PolynomialTransformer(NumericTransformer):

    def __init__(self, column_id, degree=2):
        NumericTransformer.__init__(self, column_id, "polynomial")
        self.seed = 42
        self.degree = degree
        self.model = PolynomialFeatures(self.degree)

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).T

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        self.model.fit(column_data)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).T

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        print column_data.shape

        return np.matrix(self.model.transform(column_data))

    def get_feature_names(self, dataset):
        internal_names = []

        for class_i in range(self.degree+1):
                internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + self.name + "_" + str(class_i))

        return internal_names
