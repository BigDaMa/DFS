from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SmoteTransformation(BaseEstimator, TransformerMixin):
    def __init__(self):
        name = 'Smote'
        self.smote = SMOTE()
        self.X = None
        self.X_resampled = None
        self.y_resampled = None

    def fit(self, X, y=None):
        X_new, y_new = self.smote.fit_resample(X, y)
        #print(X_new)
        X.resize((X_new.shape[0], X.shape[1]))
        print(X)
        X[len(X):, :] = X_new[len(X):, :]
        return self

    def transform(self, X):
        return X