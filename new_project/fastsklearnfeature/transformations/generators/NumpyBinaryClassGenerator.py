from fastsklearnfeature.transformations.binary.NonCommutativeBinaryTransformation import NonCommutativeBinaryTransformation
import numpy as np
from typing import List

class NumpyBinaryClassGenerator:
    def __init__(self, methods=[np.divide, np.subtract, np.power]):
        self.methods = methods

    def produce(self):
        transformation_classes: List[NonCommutativeBinaryTransformation] = []
        for m in self.methods:
            transformation_classes.append(NonCommutativeBinaryTransformation(m))
        return transformation_classes
