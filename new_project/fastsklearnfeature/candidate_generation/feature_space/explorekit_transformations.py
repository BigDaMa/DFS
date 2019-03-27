from typing import List
import numpy as np
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastsklearnfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from fastsklearnfeature.transformations.generators.NumpyBinaryClassGenerator import sympy_divide
from fastsklearnfeature.transformations.generators.NumpyBinaryClassGenerator import sympy_subtract
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import GroupByThenGenerator
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation

from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenmax
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenmin
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenmean
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenstd
import sympy

def get_transformation_for_feature_space():
    unary_transformations: List[UnaryTransformation] = []
    unary_transformations.append(PandasDiscretizerTransformation(number_bins=10))
    unary_transformations.append(MinMaxScalingTransformation())

    binary_transformations: List[Transformation] = []
    binary_transformations.extend(HigherOrderCommutativeClassGenerator(2,
                                                                       methods=[np.nansum, np.nanprod],
                                                                       sympy_methods=[sympy.Add, sympy.Mul]).produce())
    binary_transformations.extend(NumpyBinaryClassGenerator(methods=[np.divide, np.subtract],
                                                            sympy_methods=[sympy_divide, sympy_subtract]).produce())

    binary_transformations.extend(GroupByThenGenerator(2, methods=[np.nanmax,
                                                                   np.nanmin,
                                                                   np.nanmean,
                                                                   np.nanstd],
                                                          sympy_methods = [groupbythenmax,
                                                                           groupbythenmin,
                                                                           groupbythenmean,
                                                                           groupbythenstd]
                                                       ).produce())

    return unary_transformations, binary_transformations