from fastsklearnfeature.transformations.Transformation import Transformation
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
import numpy_indexed as npi
from typing import List, Dict, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import sympy

class FastGroupByThenTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, method, sympy_method):
        self.method = method
        self.sympy_method = sympy_method
        Transformation.__init__(self, 'GroupByThen' + self.method.__name__,
                 number_parent_features=2,
                 output_dimensions=1,
                 parent_feature_order_matters=True,
                 parent_feature_repetition_is_allowed=False)

    #0th feature will be aggregated, 1th-nth = key attributes
    def fit(self, X, y=None):
        final_mapping = dict(npi.group_by(keys=X[:, 1], values=X[:, 0].astype(np.float64), reduction=self.method))

        self.keys = list(final_mapping.keys())
        self.values = list(final_mapping.values())

        return self

    def transform(self, X):
        remapped_a = npi.remap(X[:, 1], self.keys, self.values).reshape(-1, 1).astype(np.float64)
        return remapped_a

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        #TODO: check whether feature_combination[0] has only distinct values

        #we handle conditional idempotence via sympy

        #the aggregated column has to be numeric
        if 'float' in str(feature_combination[0].properties['type']) \
            or 'int' in str(feature_combination[0].properties['type']) \
            or 'bool' in str(feature_combination[0].properties['type']):

            return True

        return False

    def get_combinations(self, features):
        #self.parent_feature_order_matters and not self.parent_feature_repetition_is_allowed:

        iterable_collection = []
        for i in range(2, self.number_parent_features+1):
            iterable_collection.append(itertools.permutations(features, r=i))

        return itertools.chain(*iterable_collection)

    def get_name(self, candidate_feature_names):
        return "(" + self.method.__name__ + "(" + str(candidate_feature_names[0]) + ") GroupyBy " + str(candidate_feature_names[1]) + ")"

    def get_sympy_representation(self, input_attributes):
        return self.sympy_method(input_attributes[0], input_attributes[1])



