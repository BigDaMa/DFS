from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
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

group = FastGroupByThenTransformation(np.max)

group.fit(input)

print(group.transform(input))