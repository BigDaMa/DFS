import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.AllTransformer import AllTransformer

class LengthCountTransformer(AllTransformer):

    def __init__(self, column_id):
        AllTransformer.__init__(self, column_id, "str_length", 1)

        def str_length(mystring):
            return len(str(mystring))

        self.str_length = np.vectorize(str_length, otypes=[np.int])


    def transform(self, dataset, ids):
        data = np.array(self.str_length(dataset.values[ids, self.column_id]))

        return data.reshape(-1, 1)
