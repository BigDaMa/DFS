from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.RadiansTransformer import RadiansTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class RadiansMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, RadiansTransformer)
