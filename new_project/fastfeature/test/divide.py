import numpy as np
data = np.matrix([[2,2], [1, 1], [3,3]])

print(np.divide(data[0], data[1]))


print(np.mean(data, axis=1))