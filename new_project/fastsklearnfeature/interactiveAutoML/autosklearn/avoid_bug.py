import pickle
from sklearn.impute import SimpleImputer
import numpy as np

from fastsklearnfeature.transformations.MyImputationTransformation import ImputationTransformation

from numpy import inf

f = open('/tmp/data.pickle', 'rb')

X = pickle.load(f)

'''
X[X == -np.inf] = 0
X[X == np.inf] = 0


X[np.isneginf(X)] = 0
X[np.isinf(X)] = 0

print(X)

print(np.isinf(X).any())
print(np.isfinite(X).all())
'''


t = ImputationTransformation()
t.fit(X)

print(t.transform(X))

