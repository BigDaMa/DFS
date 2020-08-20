from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FrequencyEncodingOptuna(BaseEstimator, TransformerMixin):
    def __init__(self, relative=True):
        self.relative = relative
        super(FrequencyEncodingOptuna, self).__init__()

    def init_hyperparameters(self, trial, X, y):
        pass

    def transform(self, X):
        X_encoded = np.zeros(X.shape)
        for col in range(X_encoded.shape[1]):
            for row in range(X_encoded.shape[0]):
                try:
                    X_encoded[row, col] = self.map_value2count[col][X[row, col]]
                except:
                    pass
        return X_encoded

    def calculate_frequency(self, matrix, relative=True):
        storage = {}
        for col in range(matrix.shape[1]):
            storage[col] = {}
            (unique, counts) = np.unique(matrix[:, col], return_counts=True)
            for ui in range(len(unique)):
                if relative:
                    storage[col][unique[ui]] = counts[ui] / float(matrix.shape[0])
                else:
                    storage[col][unique[ui]] = counts[ui]

        return storage

    def fit(self, X, y=None):
        self.map_value2count = self.calculate_frequency(X, self.relative)
        return self

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass
