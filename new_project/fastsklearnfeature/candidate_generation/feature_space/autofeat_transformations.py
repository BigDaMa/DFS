from typing import List
import numpy as np
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastsklearnfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from fastsklearnfeature.transformations.generators.NumpyClassGeneratorInvertible import NumpyClassGeneratorInvertible


def pow3(x):
    return x ** 3


def pow_minus_1(x):
    return 1.0 / x

def get_transformation_for_feature_space():
    # AutoFeat Feature Space
    unary_transformations: List[UnaryTransformation] = []
    unary_transformations.extend(NumpyClassGeneratorInvertible(methods=[np.exp, np.log,
                                                                        np.abs,
                                                                        np.sqrt,
                                                                        np.square,
                                                                        pow3, pow_minus_1]).produce())

    binary_transformations: List[Transformation] = []
    binary_transformations.extend(
        HigherOrderCommutativeClassGenerator(2, methods=[np.nansum, np.nanprod]).produce())
    binary_transformations.extend(NumpyBinaryClassGenerator(methods=[np.subtract]).produce())

    return unary_transformations, binary_transformations