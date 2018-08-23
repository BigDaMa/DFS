import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.AllTransformer import AllTransformer


class HashingTransformer(AllTransformer):

    def __init__(self, column_id, number_features=100):
        AllTransformer.__init__(self, column_id, "hash")
        self.number_features = number_features
        self.hashing_model = HashingVectorizer(n_features=self.number_features)


    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        return self.hashing_model.transform(column_data)

    def get_feature_names(self, dataset):
        internal_names = []

        for class_i in range(self.number_features):
                internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "hash_" + str(class_i))

        return internal_names

    def __str__(self):
        return self.__class__.__name__ + "_dimensionality_" + str(self.number_features)
