from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
from typing import List
import numpy as np
import sympy
from sympy.core.numbers import NegativeOne
from fastsklearnfeature.transformations.MinMaxScalingTransformation import scale
from fastsklearnfeature.transformations.ImputationTransformation import impute

class groupbythen(sympy.Function):
    @classmethod
    def eval(cls, value, key):
        new_key = key
        new_value = value

        #transformations that do not change the number of unique values
        if isinstance(key, sympy.Mul) and key._args[0] == NegativeOne:
            new_key = key._args[1]
        if isinstance(key, sympy.Pow) and key._args[1] == NegativeOne:
            new_key = key._args[0]
        if isinstance(key, scale):
            new_key = key._args[0]
        if isinstance(key, impute):
            new_key = key._args[0]

        if new_value != value or new_key != key:
            return cls(new_value, new_key)


class groupbythenIdempotentFunction(groupbythen):
    @classmethod
    def eval(cls, value, key):
        evaluated = super(groupbythenIdempotentFunction, cls).eval(value, key)
        if not isinstance(evaluated, groupbythenIdempotentFunction) and not evaluated == None:
            return evaluated

        new_value = value
        new_key = key
        if evaluated is not None:
            new_value = evaluated._args[0]
            new_key = evaluated._args[1]

        if isinstance(new_value, groupbythen) and new_key == new_value.args[1]:  # conditional idempotent
            return new_value

        if new_value == new_key:
            return new_value

        if new_value != value or new_key != key:
            return cls(new_value, new_key)

class groupbythenmin(groupbythenIdempotentFunction):
    @classmethod
    def eval(cls, value, key):
        evaluated = super(groupbythenmin, cls).eval(value, key)
        if not isinstance(evaluated, groupbythenmin) and not evaluated == None:
            return evaluated

        new_value = value
        new_key = key
        if evaluated is not None:
            new_value = evaluated._args[0]
            new_key = evaluated._args[1]

        if isinstance(new_value, sympy.Mul) and new_value._args[0] == NegativeOne:
            new_value = new_value._args[1]
            return -1 * groupbythenmax(new_value, new_key)

        if new_value != value or new_key != key:
            return cls(new_value, new_key)

class groupbythenmax(groupbythenIdempotentFunction):
    @classmethod
    def eval(cls, value, key):
        evaluated = super(groupbythenmax, cls).eval(value, key)
        if not isinstance(evaluated, groupbythenmax) and not evaluated == None:
            return evaluated

        new_value = value
        new_key = key
        if evaluated is not None:
            new_value = evaluated._args[0]
            new_key = evaluated._args[1]

        if isinstance(new_value, sympy.Mul) and new_value._args[0] == NegativeOne:
            new_value = new_value._args[1]
            return -1 * groupbythenmin(new_value, new_key)

        if new_value != value or new_key != key:
            return cls(new_value, new_key)

class groupbythenmean(groupbythenIdempotentFunction):
    @classmethod
    def eval(cls, value, key):
        evaluated = super(groupbythenmean, cls).eval(value, key)
        if not isinstance(evaluated, groupbythenmean) and not evaluated == None:
            return evaluated

        new_value = value
        new_key = key
        if evaluated is not None:
            new_value = evaluated._args[0]
            new_key = evaluated._args[1]

        if isinstance(new_value, sympy.Mul) and new_value._args[0] == NegativeOne:
            new_value = new_value._args[1]
            return -1 * cls(new_value, new_key)

        if new_value != value or new_key != key:
            return cls(new_value, new_key)

class groupbythenstd(groupbythen):
    @classmethod
    def eval(cls, value, key):
        evaluated = super(groupbythenstd, cls).eval(value, key)
        if not isinstance(evaluated, groupbythenstd) and not evaluated == None:
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

        if new_value == new_key:
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