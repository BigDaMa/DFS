import numpy as np

import dateutil.parser
import time
import datetime


class DateTransformer():

    def __init__(self, column_id):
        self.column_id = column_id


    def fit(self, dataset, ids):
        #nothing
        return

    def transform(self, dataset, ids):
        column_data = dataset[dataset.columns[self.column_id]].copy()

        as_timestamp = column_data.apply(lambda x: time.mktime((dateutil.parser.parse(x)).timetuple())).values[ids]
        #print as_timestamp

        return np.matrix(as_timestamp).T

    def get_feature_names(self, dataset):
        return [(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "timestamp")]

    def get_involved_columns(self):
        return [self.column_id]