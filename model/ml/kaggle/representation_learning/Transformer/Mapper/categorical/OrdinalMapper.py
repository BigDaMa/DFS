from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.OrdinalTransformer import OrdinalTransformer
from ml.kaggle.representation_learning.Transformer.Mapper.categorical.CategoricalMapper import CategoricalMapper

class OrdinalMapper(CategoricalMapper):

    def __init__(self):
        CategoricalMapper.__init__(self, OrdinalTransformer)
