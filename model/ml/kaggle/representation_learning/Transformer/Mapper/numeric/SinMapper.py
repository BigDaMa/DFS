from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.SinTransformer import SinTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class SinMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, SinTransformer)
