from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
import sympy
from fastsklearnfeature.transformations.mdlp_discretization.MDLP import MDLP_Discretizer

class mdlpdiscretize(sympy.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, mdlpdiscretize): #idempotent
            return x


class MDLPDiscretizerTransformation(BaseEstimator, TransformerMixin, NumericUnaryTransformation):
    def __init__(self):
        name = 'mdlpdiscretize'
        self.discretizer = MDLP_Discretizer(features=[0])
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, X, y=None):
        self.discretizer.fit(X, y)
        return self

    def transform(self, X):
        return self.discretizer.transform(X)

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(MDLPDiscretizerTransformation, self).is_applicable(feature_combination):
            return False
        if isinstance(feature_combination[0].transformation, MDLPDiscretizerTransformation):
            return False
        if feature_combination[0].properties['missing_values']:
            return False

        if 'number_distinct_values' in feature_combination[0].properties and feature_combination[0].properties['number_distinct_values'] <= 2:
            return False

        return True

    def get_sympy_representation(self, input_attributes):
        return mdlpdiscretize(input_attributes[0])

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            # missing values properties
            properties['missing_values'] = parents[0].properties['missing_values']

            # range properties
            properties['has_zero'] = True
            properties['min'] = 0
            properties['max'] = len(self.discretizer._cuts[0])
        except:
            # was nonnumeric data
            pass

        if 'missing_values' in parents[0].properties:
            if parents[0].properties['missing_values'] == True:
                properties['number_distinct_values'] = len(self.discretizer._cuts[0]) + 2
            else:
                properties['number_distinct_values'] = len(self.discretizer._cuts[0]) + 1
        else:
            properties['number_distinct_values'] = len(self.discretizer._cuts[0]) + 1

        return properties