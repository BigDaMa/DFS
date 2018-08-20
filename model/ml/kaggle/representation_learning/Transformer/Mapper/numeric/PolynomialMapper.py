from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.PolynomialTransformer import PolynomialTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class PolynomialMapper(NumericMapper):

    def __init__(self, degree=2):
        NumericMapper.__init__(self, PolynomialTransformer, {'degree': degree})
