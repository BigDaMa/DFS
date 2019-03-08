from fastsklearnfeature.transformations.Transformation import Transformation
from typing import Dict, List, Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import itertools

class GroupByThenTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, method, number_parent_features):
        self.method = method
        Transformation.__init__(self, 'GroupByThen' + self.method.__name__,
                 number_parent_features,
                 output_dimensions=1,
                 parent_feature_order_matters=True,
                 parent_feature_repetition_is_allowed=False)

    #0th feature will be aggregated, 1th-nth = key attributes
    def fit(self, X, y=None):
        self.key_attributes = list(range(1, (self.number_parent_features)))

        mapping: Dict[Any, float] = {}
        for record_i in range(X.shape[0]):
            key = tuple(str(element) for element in X[record_i, self.key_attributes])
            if not key in mapping:
                mapping[key]: List[float] = []
            mapping[key].append(float(X[record_i, 0]))

        self.final_mapping = {}
        for k, v in mapping.items():
            self.final_mapping[k] = self.method(np.array(v))

        return self

    def transform(self, X):
        result = np.zeros((len(X), 1))
        for i in range(len(X)):
            key = tuple(str(element) for element in X[i, self.key_attributes])
            if key in self.final_mapping:
                result[i] = self.final_mapping[key]
        return result

    def is_applicable(self, feature_combination):
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



