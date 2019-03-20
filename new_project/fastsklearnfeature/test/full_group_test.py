from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
from fastsklearnfeature.transformations.GroupByThenTransformation import GroupByThenTransformation
import numpy as np

#input = np.random.randint(1000, size=(10000, 2))

'''
input = np.array([[1, 0],
         [2, 1],
         [3, 0],
         [4, 1],
         [5, 0],
         [6, 1]])
'''


input = np.array([[1, 'a'],
         [2, 'b'],
         [3, 'a'],
         [4, 'b'],
         [5, 'a'],
         [6, 'b']])

print(input)
print(input.shape)

fast_group = FastGroupByThenTransformation(np.max)
fast_group.fit(input)
fast_transformed = fast_group.transform(input)

group = GroupByThenTransformation(np.max, 2)
group.fit(input)
transformed = group.transform(input)

print(fast_transformed)
print(transformed)

assert np.allclose(fast_transformed, transformed), "Group != FastGroup"
