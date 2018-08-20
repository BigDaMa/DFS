from ml.kaggle.representation_learning.Transformer.Mapper.all.AllMapper import AllMapper
from kaggle.representation_learning.Transformer.TransformerImplementations.all.HashingTransformer import HashingTransformer

class HashingMapper(AllMapper):

    def __init__(self, number_features=100):
        AllMapper.__init__(self, HashingTransformer, {'number_features': number_features})
