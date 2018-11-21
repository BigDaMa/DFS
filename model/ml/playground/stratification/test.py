from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

target = np.array([0, 0, 1, 1, 1, 1])
x = np.array([2,3,3,3,3,2])

splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.5)


for train_index, test_index in splitter.split(x, target):
    print("TRAIN:", train_index, "TEST:", test_index)

    print str(x[train_index]) + " " + str(target[train_index])

    print str(x[test_index]) + " " + str(target[test_index])

    break