import pandas as pd
import numpy as np
from typing import List
import openml
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.configuration.Config import Config
import copy
import pickle

class OnlineOpenMLReader:
    def __init__(self, taskID, test_folds=1):
        self.task_id = taskID
        self.raw_features: List[RawFeature] = []
        self.test_folds = test_folds
        openml.config.apikey = Config.get('openML.apikey')

    def read(self):

        self.task = openml.tasks.get_task(self.task_id)
        dataset = openml.datasets.get_dataset(dataset_id=self.task.dataset_id)

        test_indices = np.array([])
        train_indices = np.array([])
        for fold in range(self.task.get_split_dimensions()[1]):
            _, t_indices = self.task.get_train_test_split_indices(fold=fold)

            if fold < self.test_folds:
                test_indices = np.concatenate((test_indices, t_indices), axis=None)
            else:
                train_indices = np.concatenate((train_indices, t_indices), axis=None)

        train_indices = np.array(train_indices, dtype=np.int)
        test_indices = np.array(test_indices, dtype=np.int)

        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, return_categorical_indicator=True, return_attribute_names=True)

        self.dataframe = pd.DataFrame(data=X, columns=attribute_names)

        self.splitted_values = {}
        self.splitted_target = {}

        self.splitted_target['train'] = y[train_indices]
        self.splitted_target['test'] = y[test_indices]
        self.splitted_values['train'] = X[train_indices]
        self.splitted_values['test'] = X[test_indices]

        for attribute_i in range(self.dataframe.shape[1]):
            rf = RawFeature(self.dataframe.columns[attribute_i], attribute_i, {})
            rf.derive_properties(self.dataframe[self.dataframe.columns[attribute_i]].values)
            rf.properties['categorical'] = categorical_indicator[attribute_i]
            self.raw_features.append(rf)

        return self.raw_features

