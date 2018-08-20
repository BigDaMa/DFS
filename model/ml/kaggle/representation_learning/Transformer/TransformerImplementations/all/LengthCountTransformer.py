import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class LengthCountTransformer():

    def __init__(self, column_id):
        self.column_id = column_id
        self.applicable = True

        def str_length(mystring):
            return len(str(mystring))

        self.str_length = np.vectorize(str_length, otypes=[np.int])


    def fit(self, dataset, ids):
        return

    def transform(self, dataset, ids):
        return self.str_length(dataset.values[ids, self.column_id])

    def get_feature_names(self, dataset):
        return [(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + 'str_length')]

    def get_involved_columns(self):
        return [self.column_id]
