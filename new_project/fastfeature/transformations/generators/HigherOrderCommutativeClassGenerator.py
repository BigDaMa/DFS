from fastfeature.transformations.HigherOrderCommutativeTransformation import HigherOrderCommutativeTransformation
import numpy as np
from typing import List

class HigherOrderCommutativeClassGenerator:
    def __init__(self, number_of_parents: int, methods=[np.sum, np.prod, np.mean, np.std, np.median, np.var, np.max, np.min]):
        self.number_of_parents = number_of_parents
        self.methods = methods


    def produce(self):
        transformation_classes: List[HigherOrderCommutativeTransformation] = []
        for m in self.methods:
            transformation_classes.append(HigherOrderCommutativeTransformation(m, self.number_of_parents))
        return transformation_classes
