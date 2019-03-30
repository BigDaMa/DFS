from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenstd
import sympy

res =groupbythenstd(groupbythenstd(sympy.Symbol('x'), sympy.Symbol('y')),sympy.Symbol('y'))
print(res)
print(type(res))


print(type(groupbythenstd(sympy.Symbol('x'), sympy.Symbol('y'))))