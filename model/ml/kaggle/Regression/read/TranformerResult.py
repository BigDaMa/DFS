from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ClusterTransformer import ClusterTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.CosTransformer import CosTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.IdentityTransformer import IdentityTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.LogTransformer import LogTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.ScaleTransformer import ScaleTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.SinTransformer import SinTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.SquareTransformer import SquareTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.FrequencyEncodingTransformer import FrequencyEncodingTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.OrdinalTransformer import OrdinalTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.categorical.OneHotTransformer import OneHotTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.all.HashingTransformer import HashingTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.parser.DateTransformer import DateTransformer
from kaggle.representation_learning.Transformer.TransformerImplementations.parser.LatitudeLongitudeTransformer import LatitudeLongitudeTransformer

import numpy as np

class TransformerResult():

    def __init__(self, column_id):
        self.column_id = column_id
        self.transformers = []
        self.scores = []

    def add_result(self, transformer_name, score):
        self.transformers.append(transformer_name)
        self.scores.append(score)


    def get_best_transformer(self, position_in_rank=0):
        transformer_classes = [LatitudeLongitudeTransformer,
                        DateTransformer,

                        IdentityTransformer,
                        LogTransformer,
                        SinTransformer,
                        CosTransformer,
                        SquareTransformer,
                        ScaleTransformer,
                        ClusterTransformer, #todo

                        OneHotTransformer,
                        FrequencyEncodingTransformer,
                        OrdinalTransformer,

                        HashingTransformer,
                        # SkipTransformer
                        ]

        #get best transformer name
        sorted = np.argsort(np.array(self.scores) * -1)

        #print str(np.array(self.scores)[sorted])

        name = self.transformers[sorted[position_in_rank]]

        for transformer_class in transformer_classes:
            if transformer_class.__name__ == name:
                return transformer_class(self.column_id)



