from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.PlottingPositionTransformer import PlottingPositionTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class PlottingPositionMapper(NumericMapper):

    def __init__(self):
        NumericMapper.__init__(self, PlottingPositionTransformer)
