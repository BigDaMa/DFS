from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ImputerTransformer import ImputerTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class ImputerMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, ImputerTransformer)
