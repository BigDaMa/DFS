from fastsklearnfeature.transformations.Transformation import Transformation
from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformation(BaseEstimator, TransformerMixin, Transformation):
    def __init__(self, number_parent_features):
        Transformation.__init__(self, 'identity',
                 number_parent_features, output_dimensions=number_parent_features,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)

    def transform(self, X):
        return X

    def is_applicable(self, feature_combination):
        #the aggregated column has to be numeric
        for i in range(len(feature_combination)):
            if not ('float' in str(feature_combination[i].properties['type']) \
                or 'int' in str(feature_combination[i].properties['type']) \
                or 'bool' in str(feature_combination[i].properties['type'])):
                return False
        return True

    def get_name(self, candidate_feature_names):
        slist = ''
        for name_i in candidate_feature_names:
            slist += name_i + ", "
        slist = slist[:-2]
        return '{' + slist + '}'
