import numpy as np

val = np.array([-1, 0, 1, 4, 8, 10])

res = np.radians(val)

print res

where_are_NaNs = np.isnan(res)
res[where_are_NaNs] = -1

print res