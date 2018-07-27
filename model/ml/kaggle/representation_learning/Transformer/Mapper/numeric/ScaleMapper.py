from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ScaleTransformer import ScaleTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class ScaleMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, ScaleTransformer)
