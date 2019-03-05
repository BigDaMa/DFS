import itertools
list2d = [[1,2,3],[4,5,6], [7], [8,9]]

merged = list(itertools.chain(*list2d[0:0]))

print(merged)