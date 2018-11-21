import numpy as np
import category_encoders as ce
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.categorical.CategoricalTransformer import CategoricalTransformer

class OrdinalTransformer(CategoricalTransformer):

    def __init__(self, column_id):
        CategoricalTransformer.__init__(self, column_id, "ordinal", 1)

    def fit(self, dataset, ids):
        self.encoder = ce.OrdinalEncoder()
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        self.encoder.fit(column_data)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        transformed = self.encoder.transform(column_data)
        return np.matrix(transformed.values, dtype=float)
