from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ToIntTransformer import ToIntTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class ToIntMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, ToIntTransformer)

