import pandas as pd
import numpy as np
from typing import List
import openml
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.configuration.Config import Config

class OnlineOpenMLReader:
    def __init__(self, taskID):
        self.task_id = taskID
        self.raw_features: List[RawFeature] = []
        openml.config.apikey = Config.get('openML.apikey')

    def derive_properties(self, column_id, data):
        properties = {}
        # type properties
        properties['type'] = self.dataframe.dtypes.values[column_id]

        return properties


    def read(self):

        task = openml.tasks.get_task(self.task_id)
        dataset = openml.datasets.get_dataset(dataset_id=task.dataset_id)
        train_indices, test_indices = task.get_train_test_split_indices()

        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, return_categorical_indicator=True, return_attribute_names=True)

        self.dataframe = pd.DataFrame(data=X, columns=attribute_names)

        self.splitted_values = {}
        self.splitted_target = {}

        self.splitted_target['train'] = y[train_indices]
        self.splitted_target['test'] = y[test_indices]
        self.splitted_values['train'] = X[train_indices]
        self.splitted_values['test'] = X[test_indices]

        for attribute_i in range(self.dataframe.shape[1]):
            properties = self.derive_properties(attribute_i, self.dataframe[self.dataframe.columns[attribute_i]].values)
            properties['categorical'] = categorical_indicator[attribute_i]
            self.raw_features.append(RawFeature(self.dataframe.columns[attribute_i], attribute_i, properties))


        return self.raw_features

