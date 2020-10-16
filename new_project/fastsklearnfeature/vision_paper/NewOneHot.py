from sklearn.base import BaseEstimator, TransformerMixin
import copy
import numpy as np

class NewOneHot(BaseEstimator, TransformerMixin):
    def transform(self, X):
        str_X = X.astype(str)
        new_X = np.zeros((len(X), len(self.value2id)), dtype=bool)
        for i in range(len(X)):
            if str_X[i,0] in self.value2id:
                new_X[i, self.value2id[str_X[i,0]]] = True
        return new_X

    def fit(self, X, y=None):
        str_X = X.astype(str)
        unique_values = np.unique(str_X)
        self.value2id = {}
        for uv in range(len(unique_values)):
            self.value2id[unique_values[uv]] = uv
        return self

