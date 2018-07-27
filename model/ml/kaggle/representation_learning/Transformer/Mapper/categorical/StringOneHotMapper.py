from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.OneHotTransformer import OneHotTransformer
from ml.kaggle.representation_learning.Transformer.Mapper.categorical.CategoricalMapper import CategoricalMapper

class StringOneHotMapper(CategoricalMapper):

    def __init__(self):
        CategoricalMapper.__init__(self, OneHotTransformer)
