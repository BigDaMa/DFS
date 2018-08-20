import numpy as np


class FrequencyEncodingTransformer():

    def __init__(self, column_id):
        self.column_id = column_id
        self.applicable = True


    def fit(self, dataset, ids):
        data = dataset[dataset.columns[self.column_id]].values[ids]

        self.count_dict = {}
        self.total_size = float(len(data))

        for record_i in range(len(data)):
            if data[record_i] in self.count_dict:
                self.count_dict[data[record_i]] = self.count_dict[data[record_i]] + 1
            else:
                self.count_dict[data[record_i]] = 1

    def transform(self, dataset, ids):
        data = dataset[dataset.columns[self.column_id]].values[ids]
        transformed = np.zeros((len(data), 1))
        for record_i in range(len(data)):
            if data[record_i] in self.count_dict:
                transformed[record_i, 0] = self.count_dict[data[record_i]] / self.total_size
        return np.matrix(transformed)

    def get_feature_names(self, dataset):
        return [(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "frequency")]

    def get_involved_columns(self):
        return [self.column_id]