import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.AllTransformer import AllTransformer

class SkipTransformer(AllTransformer):

    def __init__(self, column_id):
        AllTransformer.__init__(self, column_id, "skip")

    def transform(self, dataset, ids):
        return None

    def get_feature_names(self, dataset):
        return None
