from sklearn.preprocessing._discretization import KBinsDiscretizer
import numpy as np

bins = 10
d = KBinsDiscretizer(bins, encode='ordinal', strategy='uniform')

X = np.array(['hello', 'test', 'hello', 'test', 'h', 'a']).reshape(1, -1)

d.fit(X)

print(d.transform(X))