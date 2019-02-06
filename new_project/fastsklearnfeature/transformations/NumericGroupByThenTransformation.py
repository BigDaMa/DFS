from fastsklearnfeature.transformations.Transformation import Transformation
import numpy_indexed as npi
import numpy as np

class NumericGroupByThenTransformation(Transformation):
    def __init__(self, method, number_parent_features):
        self.method = method
        Transformation.__init__(self, 'GroupByThen' + self.method.__name__,
                 number_parent_features, output_dimensions=1,
                 parent_feature_order_matters=True, parent_feature_repetition_is_allowed=False)


    def fit(self, data):
        self.key_attributes = range(1,self.number_parent_features)
        grouping_result = self.method(npi.group_by(data[:, self.key_attributes]), data[:, 0])
        self.mapping = dict((str(grouping_result[0][i]), grouping_result[1][i]) for i in range(len(grouping_result[0])))


    def transform(self, data):
        result = np.zeros(len(data))
        for i in range(len(data)):
            key =str(data[i, self.key_attributes])
            if key in self.mapping:
                result[i] = self.mapping[key]
        return result

    def is_applicable(self, feature_combination):
        #the aggregated column has to be numeric
        if 'float' in str(feature_combination[0].properties['type']) \
            or 'int' in str(feature_combination[0].properties['type']) \
            or 'bool' in str(feature_combination[0].properties['type']):
            return True

        return False