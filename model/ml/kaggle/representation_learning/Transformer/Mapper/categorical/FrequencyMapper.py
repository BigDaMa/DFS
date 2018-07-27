from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.FrequencyEncodingTransformer import FrequencyEncodingTransformer
from ml.kaggle.representation_learning.Transformer.Mapper.categorical.CategoricalMapper import CategoricalMapper

class FrequencyMapper(CategoricalMapper):

    def __init__(self):
        CategoricalMapper.__init__(self, FrequencyEncodingTransformer)
