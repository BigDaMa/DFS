import itertools



my_dictionary = {'A': [0,1,2], 'B': [3,4,5], 'C': ['a']}

# {A=0, B=3}, {A=0, B=4}

my_keys = list(my_dictionary.keys())

print(list(itertools.product(*[my_dictionary[k] for k in my_keys])))

