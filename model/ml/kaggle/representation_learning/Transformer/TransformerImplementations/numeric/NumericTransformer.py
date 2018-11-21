
class NumericTransformer():

    def __init__(self, column_id, name, output_space_size=None):
        self.column_id = column_id
        self.name = name
        self.applicable = True
        self.output_space_size = output_space_size


    def fit(self, dataset, ids):
        if type(self.output_space_size) == type(None):
            self.output_space_size = 1
        return

    def fit1(self, column_data):
        if type(self.output_space_size) == type(None):
            self.output_space_size = 1
        return

    def get_feature_names(self, dataset):
        return [(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + self.name)]

    def get_involved_columns(self):
        return [self.column_id]

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        self.__str__()
