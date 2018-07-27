from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.LogTransformer import LogTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class LogMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, LogTransformer)
