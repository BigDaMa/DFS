from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.HelmertTransformer import HelmertTransformer
from ml.kaggle.representation_learning.Transformer.Mapper.categorical.CategoricalMapper import CategoricalMapper

class HelmertMapper(CategoricalMapper):

    def __init__(self):
        CategoricalMapper.__init__(self, HelmertTransformer)
