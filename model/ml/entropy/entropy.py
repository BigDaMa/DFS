import numpy as np
import random

def calc_MI(X, Y, bins):
   c_XY = np.histogram2d(X, Y, bins)[0]
   c_X = np.histogram(X, bins)[0]
   c_Y = np.histogram(Y, bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H

a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([1,1,2,2,3,3,4,4,5,5])


#random.shuffle(b)
print b


bins = 5
score = calc_MI(a, b, bins)

print score