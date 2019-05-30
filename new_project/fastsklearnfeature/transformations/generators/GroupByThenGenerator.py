#from fastsklearnfeature.transformations.GroupByThenTransformation import GroupByThenTransformation
from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
from typing import List
import numpy as np
import sympy
from sympy.core.numbers import NegativeOne

class groupbythen(sympy.Function):
    is_commutative = False
    nargs = 2

    @classmethod
    def eval(cls, value, key):
        new_key = key
        new_value = value

        if isinstance(key, sympy.Mul) and key._args[0] == NegativeOne:
            new_key = key._args[1]
        if isinstance(key, sympy.Pow) and key._args[1] == NegativeOne:
            new_key = key._args[0]

        if new_value != value or new_key != key:
            return cls(new_value, new_key)


class groupbythenIdempotentFunction(groupbythen):
    @classmethod
    def eval(cls, value, key):
        evaluated = super(groupbythenIdempotentFunction, cls).eval(value, key)

        if not isinstance(evaluated, groupbythenIdempotentFunction):
            return evaluated

        new_value = value
        new_key = key
        if evaluated is not None:
            new_value = evaluated._args[0]
            new_key = evaluated._args[1]

        if isinstance(new_value, groupbythen) and new_key == new_value.args[1]:  # conditional idempotent
            return new_value

        if new_value != value or new_key != key:
            return cls(new_value, new_key)

class groupbythenmin(groupbythenIdempotentFunction):
    nargs = 2

class groupbythenmax(groupbythenIdempotentFunction):
    nargs = 2

class groupbythenmean(groupbythenIdempotentFunction):
    nargs = 2

class groupbythenstd(groupbythen):
    @classmethod
    def eval(cls, value, key):

        evaluated = super(groupbythenstd, cls).eval(value, key)
        if not isinstance(evaluated, groupbythenstd):
            return evaluated

        new_value = value
        new_key = key
        if evaluated is not None:
            new_value = evaluated._args[0]
            new_key = evaluated._args[1]

        if isinstance(new_value, sympy.Mul) and new_value._args[0] == NegativeOne:
            new_value = new_value._args[1]

        if isinstance(new_value, groupbythen) and new_key == new_value.args[1]:  # idempotent
            return 0

        if new_value != value or new_key != key:
            return cls(new_value, new_key)


class GroupByThenGenerator:
    def __init__(self, number_of_parents: int, methods=[np.nanmean,
                                                        np.nanmax,
                                                        np.nanmin,
                                                        np.nanstd,
                                                        np.nanvar,
                                                        len,
                                                        np.nanmedian,
                                                        np.nanprod,
                                                        np.nansum
                                                        ],
                 sympy_methods=[groupbythenmean, groupbythenmax, groupbythenmin, groupbythenstd]
                 ):
        self.number_of_parents = number_of_parents
        self.methods = methods
        self.sympy_methods = sympy_methods


    def produce(self):
        transformation_classes: List[FastGroupByThenTransformation] = []
        for m in range(len(self.methods)):
            transformation_classes.append(FastGroupByThenTransformation(self.methods[m], self.sympy_methods[m]))
        return transformation_classes