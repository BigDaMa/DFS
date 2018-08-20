from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.SigmoidTransformer import SigmoidTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class SigmoidMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, SigmoidTransformer)
