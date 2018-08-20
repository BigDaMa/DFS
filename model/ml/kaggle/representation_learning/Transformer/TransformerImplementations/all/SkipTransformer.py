import numpy as np


class SkipTransformer():

    def __init__(self, column_id):
        self.column_id = column_id
        self.applicable = True


    def fit(self, dataset, ids):
        #nothing
        return

    def transform(self, dataset, ids):
        return None

    def get_feature_names(self, dataset):
        return None

    def get_involved_columns(self):
        return [self.column_id]
