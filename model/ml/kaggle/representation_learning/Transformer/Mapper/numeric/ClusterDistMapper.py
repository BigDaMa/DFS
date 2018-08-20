from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ClusterDistTransformer import ClusterDistTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class ClusterDistMapper(NumericMapper):

    def __init__(self, number_clusters=10):
        NumericMapper.__init__(self, ClusterDistTransformer, {'number_clusters': number_clusters})
