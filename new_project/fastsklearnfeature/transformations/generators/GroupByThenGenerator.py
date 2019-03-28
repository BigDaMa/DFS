#from fastsklearnfeature.transformations.GroupByThenTransformation import GroupByThenTransformation
from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
from typing import List
import numpy as np
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

class GroupByThenGenerator:
    def __init__(self, number_of_parents: int, methods=[np.mean,
                                                        np.max,
                                                        np.min,
                                                        np.std,
                                                        np.var,
                                                        len,
                                                        np.median,
                                                        np.prod,
                                                        np.sum
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