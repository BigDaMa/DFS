from sklearn.model_selection import train_test_split
import numpy as np
from ml.kaggle.datasets.data_collection import get_data
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from decimal import *

dataset = get_data()[7]
dataframe = pd.read_csv(dataset[0])

y = np.array(dataframe[dataframe.columns[dataset[1]]])

test_fraction=0.2
valid_fraction=0.2
train_fraction_final = 0.6





X_rest, X_test, y_rest, y_test = train_test_split(dataframe, y, test_size=test_fraction, stratify=y)

X_rest, X_val, y_rest, y_val = train_test_split(X_rest, y_rest, test_size=valid_fraction / (1.0 - test_fraction), stratify=y_rest)


if (train_fraction_final - (1.0 - test_fraction - valid_fraction)) < 0.0000001:
    X_train = X_rest
    y_train = y_rest
else:
    X_train, _, y_train, _ = train_test_split(X_rest, y_rest, train_size=train_fraction_final / (1.0 - test_fraction - valid_fraction), stratify=y_rest)


#ids =  X_train.index.values
ids1 = X_test.index.values
ids2 = X_val.index.values
ids3 = X_train.index.values

print ids1
print ids2
print ids3

print np.intersect1d(ids1, ids2)
print np.intersect1d(ids2, ids3)
print np.intersect1d(ids1, ids3)