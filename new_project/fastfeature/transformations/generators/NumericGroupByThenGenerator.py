from fastfeature.transformations.NumericGroupByThenTransformation import NumericGroupByThenTransformation
from typing import List
import numpy_indexed as npi
from fastfeature.transformations.add_ons.count_group import count

class GroupByThenGenerator:
    def __init__(self, number_of_parents: int, methods=[npi.GroupBy.mean,
                                                        npi.GroupBy.max,
                                                        npi.GroupBy.min,
                                                        npi.GroupBy.std,
                                                        npi.GroupBy.var,
                                                        npi.GroupBy.count,
                                                        npi.GroupBy.median,
                                                        npi.GroupBy.prod,
                                                        npi.GroupBy.sum,
                                                        npi.GroupBy.mode,
                                                        count
                                                        ]):
        self.number_of_parents = number_of_parents # Group X [1] BY Y,X,Z THEN method
        self.methods = methods


    def produce(self):
        transformation_classes: List[NumericGroupByThenTransformation] = []
        for m in self.methods:
            transformation_classes.append(NumericGroupByThenTransformation(m, self.number_of_parents))
        return transformation_classes