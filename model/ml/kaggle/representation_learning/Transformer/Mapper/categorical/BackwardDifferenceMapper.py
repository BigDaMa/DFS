from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.BackwardDifferenceTransformer import BackwardDifferenceTransformer
from ml.kaggle.representation_learning.Transformer.Mapper.categorical.CategoricalMapper import CategoricalMapper

class BackwardDifferenceMapper(CategoricalMapper):

    def __init__(self):
        CategoricalMapper.__init__(self, BackwardDifferenceTransformer)
