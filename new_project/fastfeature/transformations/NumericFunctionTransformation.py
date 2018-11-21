from fastfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation

class NumericFunctionTransformation(NumericUnaryTransformation):
    def __init__(self, math_function):
        self.math_function = math_function
        NumericUnaryTransformation.__init__(self, self.math_function.__name__)

    def transform(self, data):
        return self.math_function(data)