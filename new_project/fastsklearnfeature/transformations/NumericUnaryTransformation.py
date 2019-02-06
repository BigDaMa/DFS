from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List

class NumericUnaryTransformation(UnaryTransformation):
    def __init__(self, name):
        output_dimensions = 1
        UnaryTransformation.__init__(self, name, output_dimensions)

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(NumericUnaryTransformation, self).is_applicable(feature_combination):
            return False
        if 'float' in str(feature_combination[0].properties['type']) \
            or 'int' in str(feature_combination[0].properties['type']) \
            or 'bool' in str(feature_combination[0].properties['type']):
            return True

        return False