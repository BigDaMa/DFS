from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.TrimtailTransformer import TrimtailTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class TrimTailMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, TrimtailTransformer)
