from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.DegreesTransformer import DegreesTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class DegreesMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, DegreesTransformer)
