from fastsklearnfeature.transformations.binary.NonCommutativeBinaryTransformation import NonCommutativeBinaryTransformation
import numpy as np
from typing import List
import sympy


def sympy_divide(a, b):
    return sympy.Mul(a, sympy.Pow(b, -1))

def sympy_subtract(a, b):
    return sympy.Add(a, sympy.Mul(-1, b))


class NumpyBinaryClassGenerator:
    def __init__(self, methods=[np.divide, np.subtract, np.power],
                 sympy_methods=[sympy_divide, sympy_subtract, sympy.Pow]):
        self.methods = methods
        self.sympy_methods = sympy_methods

    def produce(self):
        transformation_classes: List[NonCommutativeBinaryTransformation] = []
        for m in range(len(self.methods)):
            transformation_classes.append(NonCommutativeBinaryTransformation(self.methods[m], self.sympy_methods[m]))
        return transformation_classes
