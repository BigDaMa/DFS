from fastfeature.transformations.BinaryTransformation import BinaryTransformation

class NonCommutativeBinaryTransformation(BinaryTransformation):
    def __init__(self, method):
        self.method = method
        BinaryTransformation.__init__(self, self.method.__name__, output_dimensions=1,
                 parent_feature_order_matters=True, parent_feature_repetition_is_allowed=False)

    def transform(self, data):
        return self.method(data[:,0], data[:,1])