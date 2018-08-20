from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.SqrtTransformer import SqrtTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class SqrtMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, SqrtTransformer)
