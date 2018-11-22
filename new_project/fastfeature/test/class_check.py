import numpy as np

#y = [True, True, False, False]

y = [True, True, False]

y = [True, False, False]

if np.sum(np.array(y)) >= 2 and np.sum(np.array(y)) <= len(y) - 2:
    print("hello")