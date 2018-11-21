import numpy as np
from sklearn import preprocessing
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.categorical.CategoricalTransformer import CategoricalTransformer

class OneHotTransformer(CategoricalTransformer):

    def __init__(self, column_id):
        CategoricalTransformer.__init__(self, column_id, "onehot")

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1

        self.value_to_id = {}
        for row_i in range(len(column_data)):
            if not column_data[row_i] in self.value_to_id:
                self.value_to_id[column_data[row_i]] = len(self.value_to_id)

        self.output_space_size = len(self.value_to_id)




    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1

        matrix = np.zeros((len(column_data), len(self.value_to_id)), dtype=bool)
        for row_i in range(len(column_data)):
            if column_data[row_i] in self.value_to_id:
                matrix[row_i, self.value_to_id[column_data[row_i]]] = True

        return matrix

    def get_feature_names(self, dataset):

        inv_map = {v: k for k, v in self.value_to_id.iteritems()}


        internal_names = []
        for class_i in range(len(self.value_to_id)):
            internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "onehot_" + str(inv_map[class_i]))

        return internal_names
