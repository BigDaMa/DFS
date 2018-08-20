from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.BinarizerTransformer import BinarizerTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class BinarizeMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, BinarizerTransformer)

