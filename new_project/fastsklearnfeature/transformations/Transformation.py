import itertools
from typing import List
import sympy
import numpy as np

class Transformation:
    """
       ----------
       name : {str}, shape (1)
           Name of the transformation
       required_properties : {dict}
           Dictionary of required properties. For instance, required_properties['missing values']=0
       """
    def __init__(self, name,
                 number_parent_features, output_dimensions=None,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False):
        self.name = name
        self.number_parent_features = number_parent_features
        self.output_dimensions = output_dimensions

        # a + b = b + a -> order does not matter
        # Group X by Y Then Max !=  Group Y by X Then Max -> order does matter
        self.parent_feature_order_matters = parent_feature_order_matters

        # a * a -> then repetition allowed
        self.parent_feature_repetition_is_allowed = parent_feature_repetition_is_allowed


    def fit(self, X, y=None):
        return self

    def transform(self, data):
        pass

    def get_sympy_representation(self, input_attributes):
        #return sympy.Function(self.name)(input_attributes[0])
        pass

    def get_name(self, candidate_feature_names):
        slist = ''
        for name_i in candidate_feature_names:
            slist += name_i + ","
        slist = slist[:-1]
        return self.name + "(" + slist + ")"

    def derive_properties(self, training_data, parents):
        properties = {}
        # type properties
        properties['type'] = training_data.dtype

        try:
            # missing values properties
            #properties['missing_values'] = np.sum(np.isnan(training_data))

            # range properties
            properties['has_zero'] = 0 in training_data
            properties['min'] = np.nanmin(training_data)
            properties['max'] = np.nanmax(training_data)
        except:
            # was nonnumeric data
            pass
        properties['number_distinct_values'] = len(np.unique(training_data))
        return properties


    def is_applicable(self, feature_combination):
        return True

    # return iterator over all possible feature combinations
    def get_combinations(self, features):
        if self.parent_feature_order_matters and self.parent_feature_repetition_is_allowed:
            return itertools.product(features, repeat=self.number_parent_features)

        if self.parent_feature_order_matters and not self.parent_feature_repetition_is_allowed:
            return itertools.permutations(features, r=self.number_parent_features)

        if not self.parent_feature_order_matters and not self.parent_feature_repetition_is_allowed:
            return itertools.combinations(features, r=self.number_parent_features)

        if not self.parent_feature_order_matters and self.parent_feature_repetition_is_allowed:
            return itertools.combinations_with_replacement(features, r=self.number_parent_features)