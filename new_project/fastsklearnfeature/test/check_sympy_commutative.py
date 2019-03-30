import sympy

print(sympy.Add(sympy.Symbol('y'), sympy.Symbol('x')))
print(sympy.Add(sympy.Symbol('x'), sympy.Symbol('y')))


class CommutativeFunction(sympy.Function):
    is_commutative = True

    @classmethod
    def eval(cls, a, b):
        if list(sympy.ordered([a, b])) == [b, a]:
            return cls(b, a)

print(CommutativeFunction(sympy.Symbol('x'), sympy.Symbol('y')))
print(CommutativeFunction(sympy.Symbol('y'), sympy.Symbol('x')))

class IdempotentFunction(sympy.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, IdempotentFunction):
            return x

class discretize(IdempotentFunction):
    nargs=1

print(discretize(discretize(sympy.Symbol('x'))))