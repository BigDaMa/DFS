from ml.kaggle.representation_learning.Transformer.Mapper.all.AllMapper import AllMapper
from kaggle.representation_learning.Transformer.TransformerImplementations.all.LengthCountTransformer import LengthCountTransformer

class LengthCountMapper(AllMapper):

    def __init__(self):
        AllMapper.__init__(self, LengthCountTransformer)
