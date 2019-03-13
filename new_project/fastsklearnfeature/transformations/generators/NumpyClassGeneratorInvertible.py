from fastsklearnfeature.transformations.NumericFunctionTransformation import NumericFunctionTransformation
import numpy as np
from scipy import stats
from typing import List

class NumpyClassGeneratorInvertible:
    def __init__(self, methods=[np.cos, np.sin, np.tan, np.cosh, np.sinh, np.tanh,
                   np.abs,
                   np.sqrt, np.square,
                   np.degrees, np.radians,
                   np.log, np.exp,
                   stats.mstats.plotting_positions, stats.mstats.rankdata,
                   stats.zscore]):

        self.methods = methods


    def produce(self):
        transformation_classes: List[NumericFunctionTransformation] = []
        for m in self.methods:
            transformation_classes.append(NumericFunctionTransformation(m))
        return transformation_classes