from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation

import numpy as np

input = np.random.randint(1000, size=(10000, 1))


p = PandasDiscretizerTransformation(number_bins=10)
s = MinMaxScalingTransformation()


first = s.fit_transform(input)
second = s.fit_transform(first)

print(np.allclose(first, second))

