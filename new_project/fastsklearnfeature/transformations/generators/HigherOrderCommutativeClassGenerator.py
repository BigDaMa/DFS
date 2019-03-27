from fastsklearnfeature.transformations.HigherOrderCommutativeTransformation import HigherOrderCommutativeTransformation
import numpy as np
from typing import List
import sympy

class HigherOrderCommutativeClassGenerator:
    def __init__(self, number_of_parents: int,
                 methods=[np.sum, np.prod, np.max, np.min], #, np.mean, np.std, np.median, np.var, np.max, np.min
                 sympy_methods=[sympy.Add, sympy.Mul, sympy.Max, sympy.Min]):
        self.number_of_parents = number_of_parents
        self.methods = methods
        self.sympy_methods = sympy_methods


    def produce(self):
        transformation_classes: List[HigherOrderCommutativeTransformation] = []
        for m in range(len(self.methods)):
            transformation_classes.append(HigherOrderCommutativeTransformation(self.methods[m], self.sympy_methods[m], self.number_of_parents))
        return transformation_classes
