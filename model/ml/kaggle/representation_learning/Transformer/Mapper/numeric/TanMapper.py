from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.TanTransformer import TanTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class TanMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, TanTransformer)
