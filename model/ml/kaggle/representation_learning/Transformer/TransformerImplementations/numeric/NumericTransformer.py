
class NumericTransformer():

    def __init__(self, column_id, name):
        self.column_id = column_id
        self.name = name
        self.applicable = True


    def fit(self, dataset, ids):
        return

    def get_feature_names(self, dataset):
        return [(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + self.name)]

    def get_involved_columns(self):
        return [self.column_id]