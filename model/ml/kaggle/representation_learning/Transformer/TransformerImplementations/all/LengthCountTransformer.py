import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class LengthCountTransformer():

    def __init__(self, column_id):
        self.column_id = column_id
        self.number_features = 100


    def fit(self, dataset, ids):
        return

    def transform(self, dataset, ids):
        '''
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        return self.hashing_model.transform(column_data)
        '''

    def get_feature_names(self, dataset):
        internal_names = []

        for class_i in range(self.number_features):
                internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "hash_" + str(class_i))

        return internal_names

    def get_involved_columns(self):
        return [self.column_id]
