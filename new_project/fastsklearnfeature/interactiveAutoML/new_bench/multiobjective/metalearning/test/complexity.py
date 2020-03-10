import numpy as np

datasets = 20
constraints = 6
D = 5
strategies = 13
max_search_time = 1

complexity = np.power(constraints,D) * datasets

print(complexity)

time = complexity * max_search_time / 24 /365
print(time)