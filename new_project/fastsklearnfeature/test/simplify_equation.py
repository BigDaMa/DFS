from sympy import simplify, cos, sin, floor, acos, asin, sqrt, power, Symbol, ceiling, Min
from sympy import Abs
from sympy import factor
from sympy.abc import x, y, a, b, c, d

from sympy import Add
from sympy import Mul
from sympy import Pow

test = 1/y


def my_division(a,b):
    return Mul(x, Pow(y,-1))


print(my_division(x,y))


import numpy as np
import time

def scale(x):
    return floor(x)




#f1= Abs(Abs(x))

#f1 = Min(Min(x,1),0)

#f1 =  Min(Min(x,y),y)

print(Mul(x,y))

# if len(a.free_symbols) == 0: Constant

#f1 = ceiling(floor(Symbol('x0'))) # Idempotence


f1 = (a * b) - (a * c) # Distributive property
f2 = a*(b - c)


tetst=a/b + 1

#f1 = (a + b) + c # Associative property
#f2 = (b + c) + a

#f1 = a / a # Allowing attribute repetition

#f1 = a+b # Commutativity
#f2 = b+a

#f1 = sqrt(a) * sqrt(a) # Inverse

print(f1)
start = time.time()
print(factor(f1))
print(time.time() - start)

s = set()
print(type(f1))
s.add(factor(f1))
s.add(f2)
print(s)


print(factor(f2))

print(simplify(f2))

