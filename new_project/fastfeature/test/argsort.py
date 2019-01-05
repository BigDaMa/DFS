import numpy as np

array = np.array([3,2,1,4,5])

ids = np.argsort(array)
print(array[ids[0]])

'''
attribute_feature_matrix = np.matrix([[1,2],[1,2]])

metafeatures = [1]

diff_matrix = np.subtract(attribute_feature_matrix[:,0:len(metafeatures)], attribute_feature_matrix[:,len(metafeatures):attribute_feature_matrix.shape[1]])

print(diff_matrix)


a = np.array([1,2,3,4,5])
b = np.array([1,2,3,4,4])

print(np.corrcoef(a,b)[0,1])


print(np.abs(np.array([-1, -2, -3])))
'''