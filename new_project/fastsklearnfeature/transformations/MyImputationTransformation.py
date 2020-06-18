from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from typing import List
import sympy
import pickle
import numpy as np
from collections import Counter
import copy

class impute(sympy.Function):
    @classmethod
    def eval(cls, value):
        if isinstance(value, impute): #idempotent
            return value

class meanimpute(impute):
    nargs = 1

class medianimpute(impute):
    nargs = 1

class mostfrequentimpute(impute):
    nargs = 1

class ImputationTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        name = self.strategy + 'Imputation'
        Transformation.__init__(self, name,
                                number_parent_features=1,
                                output_dimensions=1,
                                parent_feature_order_matters=True,
                                parent_feature_repetition_is_allowed=False)


    def get_missing_value_mask(self, X):
        nan_mask = np.isnan(X)
        inf_mask = np.isinf(X)
        ninf_mask = np.isneginf(X)

        all_mask = np.logical_or(nan_mask, inf_mask)
        all_mask = np.logical_or(all_mask, ninf_mask)

        return all_mask


    def fit(self, X, y=None):
        all_mask = np.invert(self.get_missing_value_mask(X))

        self.aggregates = np.zeros(X.shape[1])
        for c in range(X.shape[1]):
            current_column = X[all_mask[:,c], c]
            if self.strategy == 'mean':
                self.aggregates[c] = np.mean(current_column)
            if self.strategy == 'median':
                self.aggregates[c] = np.median(current_column)
            if self.strategy == 'most_frequent':
                self.aggregates[c] = Counter(current_column).most_common(1)

        return self

    def transform(self, X):
        X_new = copy.deepcopy(X)

        all_mask = self.get_missing_value_mask(X_new)
        for c in range(X.shape[1]):
            X_new[all_mask[:,c], c] = self.aggregates[c]

        return X_new



    def is_applicable(self, feature_combination: List[CandidateFeature]):
        if not super(ImputationTransformation, self).is_applicable(feature_combination):
            return False
        if isinstance(feature_combination[0].transformation, ImputationTransformation):
            return False
        if isinstance(feature_combination[0], RawFeature) and 'missing_values' in feature_combination[0].properties and feature_combination[0].properties['missing_values']:
            return True

        return False

    def get_sympy_representation(self, input_attributes):
        if self.strategy == 'mean':
            return meanimpute(input_attributes[0])
        elif self.strategy == 'median':
            return medianimpute(input_attributes[0])
        elif self.strategy == 'most_frequent':
            return mostfrequentimpute(input_attributes[0])

    def derive_properties(self, training_data, parents: List[CandidateFeature]):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            properties['missing_values'] = False

            # range properties
            properties['has_zero'] = parents[0].properties['has_zero']

            #might not work for constant
            properties['min'] = parents[0].properties['min']
            properties['max'] = parents[0].properties['max']
        except:
            # was nonnumeric data
            pass
        #approximation (could be +1)
        properties['number_distinct_values'] = parents[0].properties['number_distinct_values']
        return properties