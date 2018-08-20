from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.RSHTransformer import RSHTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class RSHMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, RSHTransformer)
