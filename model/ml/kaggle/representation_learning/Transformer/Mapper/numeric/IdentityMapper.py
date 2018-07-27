from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.IdentityTransformer import IdentityTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class IdentityMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, IdentityTransformer)

