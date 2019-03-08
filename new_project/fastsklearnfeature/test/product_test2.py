import itertools


a = ['a', 'b', 'c', 1]
b = [1,2,3]


def generate_merge(a, b, order_matters=False, repetition_allowed=False):
    #e.g. sum
    if not order_matters and repetition_allowed:
        return list(itertools.product(*[a,b]))

    # feature concat, but does not work
    if not order_matters and not repetition_allowed:
        return [[x, y] for x, y in itertools.product(*[a, b]) if x != y]


    if order_matters and repetition_allowed:
        order = set(list(itertools.product(*[a, b])))
        order = order.union(set(list(itertools.product(*[b, a]))))
        return list(order)

    # e.g. subtraction
    if order_matters and not repetition_allowed:
        order = [[x, y] for x, y in itertools.product(*[a, b]) if x != y]
        order.extend([[x, y] for x, y in itertools.product(*[b, a]) if x != y])
        return order

print(generate_merge(a,b, order_matters=True, repetition_allowed=False))

