from fastsklearnfeature.transformations.Transformation import Transformation
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
import numpy_indexed as npi
from typing import List, Dict, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import sympy
import copy

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
        self.final_mapping = dict(npi.group_by(keys=X[:, 1], values=X[:, 0].astype(np.float64), reduction=self.method))
        return self

    def transform(self, X):
        #initialize result
        n = None
        if self.method == np.nanstd:
            n = np.zeros((X.shape[0], 1))
        elif self.method == len:
            n = np.ones((X.shape[0], 1))
        else:
            n = copy.deepcopy(X[:, 0]).reshape(-1, 1).astype(np.float64)

        #apply mapping
        for k in self.final_mapping:
            n[X[:, 1] == k] = self.final_mapping[k]
        return n


    def is_applicable(self, feature_combination: List[CandidateFeature]):
        #we handle conditional idempotence via sympy

        if 'missing_values' in feature_combination[0].properties and feature_combination[0].properties['missing_values']:
            return False

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

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            # missing values properties
            properties['missing_values'] = False # for np.nanFunctions

            # range properties
            if (parents[0].properties['min'] == 0.0 and self.method == np.nanmin) or \
               (parents[0].properties['max'] == 0.0 and self.method == np.nanmax):
                properties['has_zero'] = True
            else:
                properties['has_zero'] = 0 in training_data[:,0]

            if self.method == np.nanmin:
                properties['min'] = parents[0].properties['min']
            else:
                properties['min'] = np.nanmin(training_data[:,0])


            if self.method == np.nanmax:
                properties['max'] = parents[0].properties['max']
            else:
                properties['max'] = np.nanmax(training_data[:,0])
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = parents[1].properties['number_distinct_values']
        return properties



