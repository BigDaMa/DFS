from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ZScoreTransformer import ZScoreTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class ZscoreMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, ZScoreTransformer)
