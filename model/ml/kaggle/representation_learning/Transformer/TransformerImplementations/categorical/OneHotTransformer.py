import numpy as np
from sklearn import preprocessing


class OneHotTransformer():

    def __init__(self, column_id):
        self.column_id = column_id


    def fit(self, dataset, ids):
        self.one_hot_model = preprocessing.LabelBinarizer()
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        self.one_hot_model.fit(column_data)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        return self.one_hot_model.transform(column_data)

    def get_feature_names(self, dataset):
        internal_names = []
        if len(self.one_hot_model.classes_) > 2:
            for class_i in range(len(self.one_hot_model.classes_)):
                internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "onehot_" + str(self.one_hot_model.classes_[class_i]))
        else:
            internal_names = [str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "onehot"]

        return internal_names

    def get_involved_columns(self):
        return [self.column_id]
