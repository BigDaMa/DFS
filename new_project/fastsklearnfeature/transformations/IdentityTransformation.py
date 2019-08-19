from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sympy
from sklearn.preprocessing import MinMaxScaler


class ConcatenationFunction(sympy.Function):
    def __init__(self, *args):
        self.arguments = args

    @classmethod
    def eval(cls, *args):
        def normalize_concat(inputs):
            all_arguments_stored = set()
            for input in inputs:
                if isinstance(input, ConcatenationFunction):
                    all_arguments_stored = all_arguments_stored.union(normalize_concat(input._args))
                else:
                    all_arguments_stored.add(input)
            return all_arguments_stored

        all_arguments = normalize_concat(args)
        sorted_arguments = list(sympy.ordered(all_arguments))

        if len(sorted_arguments) == 1:
            return sorted_arguments[0]

        if sorted_arguments != list(args):
            return cls(*sorted_arguments)


class IdentityTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features):
        #self.minmaxscaler = MinMaxScaler()
        Transformation.__init__(self, 'identity',
                 number_parent_features, output_dimensions=number_parent_features,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)


    def transform(self, X):
        return X

    '''
    def fit(self, X, y=None):
        try:
            self.minmaxscaler.fit(X)
        except:
            pass

        return self

    def transform(self, X):
        try:
            return self.minmaxscaler.transform(X)
        except:
            return X
    '''

    def is_applicable(self, feature_combination):
        #the aggregated column has to be numeric
        for i in range(len(feature_combination)):
            if not ('float' in str(feature_combination[i].properties['type']) \
                or 'int' in str(feature_combination[i].properties['type']) \
                or 'bool' in str(feature_combination[i].properties['type'])):
                return False

        try:
            check_missing_values = any([combo.properties['missing_values'] for combo in feature_combination])
            if check_missing_values:
                return False
        except:
            print(feature_combination)

        return True

    def get_name(self, candidate_feature_names):
        slist = ''
        for name_i in candidate_feature_names:
            slist += name_i + ", "
        slist = slist[:-2]
        return '{' + slist + '}'

    def derive_properties(self, training_data, parents):
        properties = {}
        properties['type'] = np.float
        properties['missing_values'] = False
        return properties

    def get_sympy_representation(self, input_attributes):
        return ConcatenationFunction(*input_attributes)