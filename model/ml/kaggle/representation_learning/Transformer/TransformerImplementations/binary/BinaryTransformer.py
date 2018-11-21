
class BinaryTransformer():

    def __init__(self, column_a, column_b, name, output_space_size=None):
        self.column_a = column_a
        self.column_b = column_b
        self.name = name
        self.applicable = True
        self.output_space_size = output_space_size


    def fit(self, dataset, ids):
        if type(self.output_space_size) == type(None):
            self.output_space_size = 1
        return

    def fit1(self, a, b):
        if type(self.output_space_size) == type(None):
            self.output_space_size = 1
        return

    def get_feature_names(self, dataset):
        return [(str(self.column_a) + '#' + str(dataset.columns[self.column_a]) + "#" + self.name + ' #' + str(self.column_b) + '#' + str(dataset.columns[self.column_b]))]

    def get_involved_columns(self):
        return [self.column_a, self.column_b]

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        self.__str__()
