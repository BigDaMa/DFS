import numpy as np
import re



class ParseNumbersTransformer():

    def __init__(self, column_id, max_numbers=5):
        self.column_id = column_id
        self.max_numbers = max_numbers
        self.applicable = True

        def parse_numbers(mystring):
            my_list = [int(s) for s in re.findall(r'\d+', mystring)]
            my_array = np.zeros(self.max_numbers, dtype=np.int)

            for list_i in range(len(my_list)):
                my_array[list_i] = my_list[list_i]

            return my_array

        self.parse_numbers = np.vectorize(parse_numbers, otypes=[np.ndarray])


    def fit(self, dataset, ids):
        return

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

    def get_involved_columns(self):
        return [self.column_id]
