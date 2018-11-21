import numpy as np
import re
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.AllTransformer import AllTransformer


class ParseNumbersTransformer(AllTransformer):


    def __init__(self, column_id, max_numbers=5):
        AllTransformer.__init__(self, column_id, "parse_numbers", max_numbers)
        self.max_numbers = max_numbers

        self.parse_numbers = np.vectorize(self.parse_numbers_method, otypes=[np.ndarray])

    def parse_numbers_method(self, mystring):
        my_list = [int(s) for s in re.findall(r'\d+', mystring)]
        my_array = np.zeros(self.max_numbers, dtype=np.int)

        for list_i in range(len(my_list)):
            my_array[list_i] = my_list[list_i]

        return my_array


    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        new_matrix = self.parse_numbers(column_data)
        return np.matrix(np.stack(new_matrix))


    def get_feature_names(self, dataset):
        internal_names = []

        for class_i in range(self.max_numbers):
            internal_names.append(
                str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "parse_numbers_" + str(class_i))

        return internal_names

    def __str__(self):
        return self.__class__.__name__ + "_dimensionality_" + str(self.max_numbers)
