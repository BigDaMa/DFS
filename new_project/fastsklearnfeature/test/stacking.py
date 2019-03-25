import numpy as np

a = np.ones((3,1))

b = np.array(['a', 'b', 'c']).reshape((3,1))

my_list = [a,b]

print(np.hstack(my_list))