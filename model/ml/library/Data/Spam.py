import numpy as np
from sklearn.datasets import fetch_20newsgroups
import pandas as pd


class Spam():

    def __init__(self, train_fraction=0.8):
        data = pd.read_csv("/home/felix/Downloads/spam.csv", header=0, encoding='latin-1')

        myrandom = np.random.RandomState()
        self.ids = np.arange(data.shape[0])
        myrandom.shuffle(self.ids)

        self.border = int(train_fraction * data.shape[0])

        self.x = data.values[:, 1]
        self.y = data.values[:, 0]

    def get_train(self):
        return (self.x[self.ids[0:self.border]], self.y[self.ids[0:self.border]])

    def get_test(self):
        return (self.x[self.ids[self.border:self.x.shape[0]]], self.y[self.ids[self.border:self.y.shape[0]]])

    def get_number_classes(self):
        return 2

if __name__ == "__main__":
    data = Spam()

    print data.get_test()[0].shape
