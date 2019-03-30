import sympy

class groupbythen(sympy.Function):
    is_commutative = False
    nargs = 2

class groupbythenIdempotentFunction(groupbythen):
    @classmethod
    def eval(cls, value, key):
        if isinstance(value, groupbythen) and key == value.args[1]:  # conditional idempotent
            return value

class groupbythenmin(groupbythenIdempotentFunction):
    nargs = 2

class groupbythenmax(groupbythenIdempotentFunction):
    nargs = 2

class groupbythenmean(groupbythenIdempotentFunction):
    nargs = 2

class groupbythenstd(groupbythen):
    @classmethod
    def eval(cls, value, key):
        if isinstance(value, groupbythen) and key == value.args[1]:  # idempotent
            return 0


x = sympy.Symbol('x')
y = sympy.Symbol('y')
z = sympy.Symbol('z')
a = sympy.Symbol('a')

print(groupbythenstd(groupbythenmin(x,y),y))
print(groupbythenmin(groupbythenmin(x,a+y),y+a))

print(groupbythenmean(groupbythenmax(groupbythenmin(x,y),y),y))
print(groupbythenstd(groupbythenmean(groupbythenmax(groupbythenmin(x,y),y),y), y))


test = groupbythenstd(groupbythenmean(groupbythenmax(groupbythenmin(x,y),y),y), y)