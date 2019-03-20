#from fastsklearnfeature.transformations.GroupByThenTransformation import GroupByThenTransformation
from fastsklearnfeature.transformations.FastGroupByThenTransformation import FastGroupByThenTransformation
from typing import List
import numpy as np

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
                                                        ]):
        self.number_of_parents = number_of_parents # Group X [1] BY Y,X,Z THEN method
        self.methods = methods


    def produce(self):
        transformation_classes: List[FastGroupByThenTransformation] = []
        for m in self.methods:
            transformation_classes.append(FastGroupByThenTransformation(m))
        return transformation_classes