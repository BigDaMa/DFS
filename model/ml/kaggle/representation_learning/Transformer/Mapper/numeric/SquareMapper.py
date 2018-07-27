from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.SquareTransformer import SquareTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class SquareMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, SquareTransformer)
