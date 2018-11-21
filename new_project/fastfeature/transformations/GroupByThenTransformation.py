from fastfeature.transformations.Transformation import Transformation
from typing import Dict, List, Any
import numpy as np

class GroupByThenTransformation(Transformation):
    def __init__(self, method, number_parent_features):
        self.method = method
        Transformation.__init__(self, 'GroupByThen' + self.method.__name__,
                 number_parent_features, output_dimensions=1,
                 parent_feature_order_matters=True, parent_feature_repetition_is_allowed=False)

    def fit(self, data):
        self.key_attributes = range(1, self.number_parent_features)

        mapping: Dict[Any, float] = {}
        for record_i in range(data.shape[0]):
            key = tuple(str(element) for element in data[record_i, self.key_attributes])
            if not key in mapping:
                mapping[key]: List[float] = []
            mapping[key].append(float(data[record_i, 0]))

        self.final_mapping = {}
        for k, v in mapping.items():
            self.final_mapping[k] = self.method(np.array(v))


    def transform(self, data):
        result = np.zeros(len(data))
        for i in range(len(data)):
            key = tuple(element for element in data[i, self.key_attributes])
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