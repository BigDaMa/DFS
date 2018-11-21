import itertools

features = [1,2,3,4,5]
number_parents = 2


for e in itertools.product(features, repeat=number_parents): # order matters, with repetition
    print(e)

#for e in itertools.permutations(features, r=number_parents): # order matters, no repetition
#    print(e)

#for e in itertools.combinations(features, r=number_parents): # order does not matters, no repetition
#    print(e)

#for e in itertools.combinations_with_replacement(features, r=number_parents): # order does not matters, with repetition
#    print(e)