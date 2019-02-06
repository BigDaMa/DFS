from fastsklearnfeature.transformations.Transformation import Transformation

class BinaryTransformation(Transformation):
    def __init__(self, name, output_dimensions=None,
                 parent_feature_order_matters=False, parent_feature_repetition_is_allowed=False):
        number_parent_features = 2
        Transformation.__init__(self, name,
                                number_parent_features, output_dimensions,
                                parent_feature_order_matters, parent_feature_repetition_is_allowed)