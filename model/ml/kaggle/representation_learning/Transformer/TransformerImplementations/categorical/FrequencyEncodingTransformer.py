import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.categorical.CategoricalTransformer import CategoricalTransformer

class FrequencyEncodingTransformer(CategoricalTransformer):

    def __init__(self, column_id):
        CategoricalTransformer.__init__(self, column_id, "frequency", 1)


    def fit(self, dataset, ids):
        data = dataset[dataset.columns[self.column_id]].values[ids]
        self.fit1(data)

    def fit1(self, data):
        self.count_dict = {}
        self.total_size = float(len(data))

        for record_i in range(len(data)):
            if data[record_i] in self.count_dict:
                self.count_dict[data[record_i]] = self.count_dict[data[record_i]] + 1
            else:
                self.count_dict[data[record_i]] = 1



    def transform1(self, data):
        transformed = np.zeros((len(data), 1))
        for record_i in range(len(data)):
            if data[record_i] in self.count_dict:
                transformed[record_i, 0] = self.count_dict[data[record_i]] / self.total_size
        return np.matrix(transformed)

    def transform(self, dataset, ids):
        column_data = dataset[dataset.columns[self.column_id]].values[ids]

        return self.transform1(column_data)