from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.DummyTransformer import DummyTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class DummyMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, DummyTransformer)
