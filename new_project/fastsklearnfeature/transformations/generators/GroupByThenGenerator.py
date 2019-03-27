#from fastsklearnfeature.transformations.GroupByThenTransformation import GroupByThenTransformation
from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
from typing import List
import numpy as np
import sympy

class groupbythenmean(sympy.Function):
    nargs = (1,2)
    is_commutative = False

class groupbythenmax(sympy.Function):
    nargs = (1,2)
    is_commutative = False

class groupbythenmin(sympy.Function):
    nargs = (1,2)
    is_commutative = False

class groupbythenstd(sympy.Function):
    nargs = (1,2)
    is_commutative = False

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
        self.number_of_parents = number_of_parents # Group X [1] BY Y,X,Z THEN method
        self.methods = methods
        self.sympy_methods = sympy_methods


    def produce(self):
        transformation_classes: List[FastGroupByThenTransformation] = []
        for m in range(len(self.methods)):
            transformation_classes.append(FastGroupByThenTransformation(self.methods[m], self.sympy_methods[m]))
        return transformation_classes