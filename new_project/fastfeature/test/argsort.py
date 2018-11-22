import numpy as np

array = np.array([3,2,1,4,5])

ids = np.argsort(array)
print(array[ids[0]])