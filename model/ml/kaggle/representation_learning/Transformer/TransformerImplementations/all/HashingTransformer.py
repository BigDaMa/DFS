import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class HashingTransformer():

    def __init__(self, column_id, number_features=100):
        self.column_id = column_id
        self.number_features = number_features
        self.applicable = True


    def fit(self, dataset, ids):
        self.hashing_model = HashingVectorizer(n_features=self.number_features)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        return self.hashing_model.transform(column_data)

    def get_feature_names(self, dataset):
        internal_names = []

        for class_i in range(self.number_features):
                internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "hash_" + str(class_i))

        return internal_names

    def get_involved_columns(self):
        return [self.column_id]
