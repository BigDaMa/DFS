import numpy as np

arr = np.array([[1,2,'a'],
                [2,2,'a'],
                [2,2,'b'],
                [3,3,'b'],
                ])



import numpy_indexed as npi

result = (npi.group_by(arr[:, [1,2]]).mean(arr[:,0]) )

#print((str(result[0][i]), result[1][i]) for i in range(len(result)))


mapping = dict((str(result[0][i]), result[1][i]) for i in range(len(result[0])))
#print(mapping)

npi.GroupBy.mean

method = npi.GroupBy.mean

print(method.__name__)

result = (method(npi.group_by(arr[:, [1,2]]),arr[:,0]) )
print(result)