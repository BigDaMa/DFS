from fastfeature.transformations.NumericFunctionTransformation import NumericFunctionTransformation
import numpy as np
from scipy import stats
from typing import List

class NumpyClassGenerator:
    def __init__(self, methods=[np.square, np.sqrt, np.abs, np.rint, stats.mstats.rsh, stats.mstats.trimtail, stats.mstats.winsorize]):
        self.methods = methods


    def produce(self):
        transformation_classes: List[NumericFunctionTransformation] = []
        for m in self.methods:
            transformation_classes.append(NumericFunctionTransformation(m))
        return transformation_classes
