from kaggle.representation_learning.Transformer.TransformerImplementations.all.SkipTransformer import SkipTransformer
from ml.kaggle.representation_learning.Transformer.Mapper.all.AllMapper import AllMapper

class SkipMapper(AllMapper):

    def __init__(self):
        AllMapper.__init__(self, SkipTransformer)
