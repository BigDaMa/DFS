from ml.kaggle.representation_learning.Transformer.Mapper.all.AllMapper import AllMapper
from kaggle.representation_learning.Transformer.TransformerImplementations.all.ParseNumbersTransformer import ParseNumbersTransformer

class ParseNumbersMapper(AllMapper):

    def __init__(self, max_numbers=5):
        AllMapper.__init__(self, ParseNumbersTransformer, {'max_numbers': max_numbers})
