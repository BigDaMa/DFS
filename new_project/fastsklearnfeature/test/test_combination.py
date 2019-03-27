import itertools
import time

a = [1,2,3]
b = [1,2,3]

#print(list(itertools.product(*[a, b])))

print(set([frozenset([x, y]) if x != y else (x, y) for x, y in itertools.product(*[a, b])]))

start = time.time()
for i in range(10000):
    t = set([frozenset([x, y]) for x, y in itertools.product(*[a, b]) if x != y])
print(time.time() - start)


#order = set(list(itertools.product(*[a, b])))

#order_matters and not repetition_allowed:
order = set([(x, y) for x, y in itertools.product(*[a, b]) if x != y])
order = order.union([(x, y) for x, y in itertools.product(*[b, a]) if x != y])
#print(order)


#print([[x, y] for x, y in itertools.product(*[a, b]) if x != y])

#order_matters and repetition_allowed:
order = set(list(itertools.product(*[a, b])))
order = order.union(set(list(itertools.product(*[b, a]))))
#print(order)
