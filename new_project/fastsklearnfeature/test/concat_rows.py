import numpy as np

a = np.ones((3,3))

b = np.zeros((5,3))

my_list = [a,b]

print(np.vstack(my_list))