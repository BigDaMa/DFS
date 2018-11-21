from fastfeature.transformations.Transformation import Transformation

class UnaryTransformation(Transformation):
    def __init__(self, name, output_dimensions=None):
        number_parent_features = 1
        parent_feature_order_matters = False
        parent_feature_repetition_is_allowed = False

        Transformation.__init__(self, name,
                                number_parent_features, output_dimensions,
                                parent_feature_order_matters, parent_feature_repetition_is_allowed)

    def is_applicable(self, feature_combination):
        if len(feature_combination) != 1:
            return False
        return True