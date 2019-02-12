import itertools


a = ['a', 'b', 'c']
b = [1,2,3]
c = [True, False]

print(list(itertools.product (*[a,b,c])))