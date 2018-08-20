from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.QuantileTransformer import QuantileTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class QuantileMapper(NumericMapper):

    def __init__(self, output_distribution='normal'):
        NumericMapper.__init__(self, QuantileTransformer, {'output_distribution': output_distribution})
