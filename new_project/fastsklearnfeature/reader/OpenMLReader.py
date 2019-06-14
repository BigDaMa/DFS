import pandas as pd
import numpy as np
from typing import List
import json

from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.splitting.Splitter import Splitter
from fastsklearnfeature.configuration.Config import Config

class OpenMLReader:
    def __init__(self, name, splitter):
        self.name: str = name
        self.raw_features: List[RawFeature] = []
        self.splitter: Splitter = splitter

    def read(self):
        openML_path = Config.get('openml.path')

        info_frame = pd.read_csv(openML_path + "/info.csv")

        assert info_frame[info_frame['name'] == self.name]['MLType'].values == 'classification', "it is not a classification task"


        #get schema and target
        file = open(openML_path + "/data/" + self.name + "_columns.csv", mode='r')
        json_schema = file.read()
        file.close()
        schema = json.loads(json_schema)

        names = [s['name']  for s in schema]

        self.dataframe = pd.read_csv(openML_path + "/data/" + self.name + ".csv", )

        self.target_column_id = np.where(self.dataframe.columns == 'target')[0][0]


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
    s = Splitter()

    r = OpenMLReader("GCM", s)
    r.read()

    print(str(r.raw_features))