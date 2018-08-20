from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.WinsorizeTransformer import WinsorizeTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class WinsorizeMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, WinsorizeTransformer)
