from fastsklearnfeature.transformations.Transformation import Transformation
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
import numpy_indexed as npi


class FastGroupByThenTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, method):
        self.method = method
        Transformation.__init__(self, 'GroupByThen' + self.method.__name__,
                 number_parent_features=2,
                 output_dimensions=1,
                 parent_feature_order_matters=True,
                 parent_feature_repetition_is_allowed=False)

    #0th feature will be aggregated, 1th-nth = key attributes
    def fit(self, X, y=None):
        #final_mapping = dict(npi.group_by(keys=X[:, 1], values=X[:, 0].astype(np.float64), reduction=self.method))
        final_mapping = dict(npi.group_by(keys=X[:, 1], values=X[:, 0], reduction=self.method))

        self.keys = list(final_mapping.keys())
        self.values = list(final_mapping.values())

        return self

    def transform(self, X):
        #remapped_a = npi.remap(X[:, 1], self.keys, self.values).reshape(-1, 1).astype(np.float64)
        remapped_a = npi.remap(X[:, 1], self.keys, self.values).reshape(-1, 1)
        return remapped_a

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



