from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.CosTransformer import CosTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class CosMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, CosTransformer)
