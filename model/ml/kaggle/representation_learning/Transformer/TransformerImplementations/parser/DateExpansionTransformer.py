import numpy as np

import dateutil.parser
import time
import datetime


class DateExpansionTransformer():
    # date -> sec, min, day, weekday, month, year

    def __init__(self, column_id):
        self.column_id = column_id


    def fit(self, dataset, ids):
        #nothing
        return

    def transform(self, dataset, ids):
        '''
        '''

    def get_feature_names(self, dataset):
        return [(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "timestamp")]

    def get_involved_columns(self):
        return [self.column_id]