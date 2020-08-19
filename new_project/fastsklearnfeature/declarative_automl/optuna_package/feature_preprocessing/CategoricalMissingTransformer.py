from sklearn.base import BaseEstimator, TransformerMixin
import copy
import numpy as np

class CategoricalMissingTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        if str(X.dtype) == 'float32':
            new_X = copy.deepcopy(X).astype('|S10')
            #print(new_X)
            #new_X[np.isnan(new_X)] = 'MissingValueNan'
            #print(new_X)
            return new_X
        return X

    def fit(self, X, y=None):
        return self

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass
