from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.RankTransformer import RankTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class RankMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, RankTransformer)
