import numpy as np

period = np.arange(10)


print(np.sum(period > 5) / float(len(period)))