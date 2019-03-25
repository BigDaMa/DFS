import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class SourceTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, X):
        self.X = X

    def transform(self, X):
        return self.X

a = np.ones((3,1))

b = np.array(['a', 'b', 'c']).reshape((3,1))

my_list = [a,b]

print(np.hstack(my_list))


data = np.rec.fromarrays(my_list)

print(SourceTransformation(a).transform(None))

print(FeatureUnion([('1', SourceTransformation(a)), ('2', SourceTransformation(b))]).transform(None))
