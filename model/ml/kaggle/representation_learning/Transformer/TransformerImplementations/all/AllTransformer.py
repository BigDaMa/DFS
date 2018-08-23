

class AllTransformer():

    def __init__(self, column_id, name):
        self.column_id = column_id
        self.name = name
        self.applicable = True


    def fit(self, dataset, ids):
        pass

    def get_feature_names(self, dataset):
        return [(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + self.name)]

    def get_involved_columns(self):
        return [self.column_id]

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        self.__str__()
