import pandas as pd
import numpy as np
from typing import List

from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.splitting.Splitter import Splitter

class Reader:
    def __init__(self, file_name, target_column_id, splitter):
        self.file_name: str = file_name
        self.target_column_id: int = target_column_id
        self.raw_features: List[RawFeature] = []
        self.splitter: Splitter = splitter

    def derive_properties(self, column_id, data):
        properties = {}
        # type properties
        properties['type'] = self.dataframe.dtypes.values[column_id]

        return properties


    def read(self):
        self.dataframe = pd.read_csv(self.file_name, na_filter=False)

        # get target
        self.target_values = self.dataframe[self.dataframe.columns[self.target_column_id]].values
        self.dataframe.drop(self.dataframe.columns[self.target_column_id], axis=1, inplace=True)

        # get split of the data
        self.splitter.get_splitted_ids(self.dataframe, self.target_values)

        self.splitted_values = {}
        self.splitted_target= {}

        self.splitted_target['train'], self.splitted_target['valid'], self.splitted_target['test'] = self.splitter.materialize_target(self.target_values)
        self.splitted_values['train'], self.splitted_values['valid'],self.splitted_values['test'] = self.splitter.materialize_values(self.dataframe)

        for attribute_i in range(self.dataframe.shape[1]):
            properties = self.derive_properties(attribute_i, self.dataframe[self.dataframe.columns[attribute_i]].values)
            self.raw_features.append(RawFeature(self.dataframe.columns[attribute_i], attribute_i, properties))

        return self.raw_features





if __name__ == '__main__':
    from fastfeature.splitting.Splitter import Splitter

    s = Splitter()

    dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13)
    r = Reader(dataset[0], dataset[1], s)
    r.read()

    print(str(r.raw_features))
