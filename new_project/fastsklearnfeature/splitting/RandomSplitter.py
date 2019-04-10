from sklearn.model_selection import train_test_split
import pandas as pd

class RandomSplitter:
    def __init__(self, train_fraction=[0.6, 1000], valid_fraction=0.2, test_fraction=0.2, seed=None):
        self.train_fraction = train_fraction
        self.valid_fraction = valid_fraction
        self.test_fraction = test_fraction
        self.seed = seed


    def get_splitted_ids(self, dataset: pd.DataFrame, y):

        train_fraction_final = -1
        if self.train_fraction[0] * len(y) > self.train_fraction[1]:
            train_fraction_final = self.train_fraction[1] / float(len(y))
        else:
            train_fraction_final = self.train_fraction[0]

        self.ids = {}
        self.ids['valid'] = []

        #print(dataset)
        X_rest, X_test, y_rest, y_test = train_test_split(dataset, y, test_size=self.test_fraction,
                                                          random_state=self.seed)
        self.ids['test'] = X_test.index.values

        if self.valid_fraction > 0.0:
            X_rest, X_val, y_rest, y_val = train_test_split(X_rest, y_rest,
                                                            test_size=self.valid_fraction / (1.0 - self.test_fraction),
                                                            random_state=self.seed)
            self.ids['valid'] = X_val.index.values

        if (train_fraction_final - (1.0 - self.test_fraction - self.valid_fraction)) < 0.1:
            X_train = X_rest
            y_train = y_rest
        else:
            X_train, _, y_train, _ = train_test_split(X_rest, y_rest, train_size=train_fraction_final / (
                        1.0 - self.test_fraction - self.valid_fraction), random_state=self.seed)

        self.ids['train'] = X_train.index.values

        return self.ids

    def materialize(self, dataset, column_id):
        values = {}
        for k,v in self.ids.items():
            values[k] = dataset[dataset.columns[column_id]].values[v]
        return values

    def materialize_values(self, dataset):
        return dataset.values[self.ids['train'],:], dataset.values[self.ids['valid'],:], dataset.values[self.ids['test'],:]

    def materialize_target(self, y):
        return y[self.ids['train']], y[self.ids['valid']], y[self.ids['test']]