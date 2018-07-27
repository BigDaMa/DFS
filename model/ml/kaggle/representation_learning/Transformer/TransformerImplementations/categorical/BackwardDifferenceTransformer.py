import numpy as np
import category_encoders as ce

class BackwardDifferenceTransformer():

    def __init__(self, column_id):
        self.column_id = column_id


    def fit(self, dataset, ids):
        self.encoder = ce.BackwardDifferenceEncoder()
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        self.encoder.fit(column_data)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        transformed = self.encoder.transform(column_data)
        self.transformed_size = transformed.shape[1]
        return transformed

    def get_feature_names(self, dataset):
        internal_names = []
        for class_i in range(self.transformed_size):
            internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "BackwardDifference_" + str(class_i))

        return internal_names

    def get_involved_columns(self):
        return [self.column_id]
