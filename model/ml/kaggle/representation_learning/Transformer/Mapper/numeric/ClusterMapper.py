from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ClusterTransformer import ClusterTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class ClusterMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, ClusterTransformer)
