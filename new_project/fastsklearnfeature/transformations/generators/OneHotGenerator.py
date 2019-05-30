from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
import numpy as np
from typing import List
from fastsklearnfeature.candidates.RawFeature import RawFeature

class OneHotGenerator:
    def __init__(self, training_all, raw_features: List[RawFeature]):
        self.training_all = training_all
        self.raw_features = raw_features

    def produce(self):
        transformation_classes: List[OneHotTransformation] = []
        for c in range(len(self.raw_features)):
            if self.raw_features[c].properties['type'] == np.dtype('O') or \
                ('categorical' in self.raw_features[c].properties and self.raw_features[c].properties['categorical']):
                distinct_values = list(np.unique(self.training_all[:, c]))
                for dv_i in range(len(distinct_values)):
                    transformation_classes.append(OneHotTransformation(distinct_values[dv_i], dv_i, self.raw_features[c]))


        return transformation_classes
