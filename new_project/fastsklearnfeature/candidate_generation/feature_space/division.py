from typing import List
import numpy as np
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import GroupByThenGenerator
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation

from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenmax
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenmin
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenmean
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import groupbythenstd
import sympy
from fastsklearnfeature.transformations.generators.OneHotGenerator import OneHotGenerator
from fastsklearnfeature.transformations.OneDivisionTransformation import OneDivisionTransformation
from fastsklearnfeature.transformations.MinusTransformation import MinusTransformation
from fastsklearnfeature.transformations.LogTransformation import LogTransformation

from fastsklearnfeature.transformations.ImputationTransformation import ImputationTransformation
from fastsklearnfeature.transformations.mdlp_discretization.MDLPDiscretizerTransformation import MDLPDiscretizerTransformation

def get_transformation_for_division(train_X_all, raw_features):

    unary_transformations: List[UnaryTransformation] = []
    binary_transformations: List[Transformation] = []

    #unary_transformations.append(PandasDiscretizerTransformation(number_bins=10))
    unary_transformations.append(MinMaxScalingTransformation())
    unary_transformations.append(MDLPDiscretizerTransformation())

    unary_transformations.append(OneDivisionTransformation())
    unary_transformations.append(MinusTransformation())
    unary_transformations.append(LogTransformation())

    unary_transformations.append(ImputationTransformation('mean'))
    #unary_transformations.append(ImputationTransformation('median'))
    #unary_transformations.append(ImputationTransformation('most_frequent'))

    
    binary_transformations.extend(HigherOrderCommutativeClassGenerator(2,
                                                                       methods=[np.nansum, np.nanprod],
                                                                       sympy_methods=[sympy.Add, sympy.Mul]).produce())

    binary_transformations.extend(GroupByThenGenerator(2, methods=[np.nanmax,
                                                                   np.nanmin,
                                                                   np.nanmean,
                                                                   np.nanstd],
                                                          sympy_methods = [groupbythenmax,
                                                                           groupbythenmin,
                                                                           groupbythenmean,
                                                                           groupbythenstd]
                                                       ).produce())

    unary_transformations.extend(OneHotGenerator(train_X_all, raw_features).produce())

    return unary_transformations, binary_transformations