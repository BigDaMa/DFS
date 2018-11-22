from fastfeature.transformations.Transformation import Transformation

class HigherOrderCommutativeTransformation(Transformation):
    def __init__(self, method, number_parent_features):
        self.method = method
        Transformation.__init__(self, self.method.__name__,
                 number_parent_features, output_dimensions=1,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False)


    def transform(self, data):
        try:
            return self.method(data, axis=1)
        except Exception as e:
            print('HigherOrderCommutativeTransformation' + str(e))
