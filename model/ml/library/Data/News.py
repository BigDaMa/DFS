import numpy as np
from sklearn.datasets import fetch_20newsgroups

class News():

    def __init__(self):
        self.newsgroups_train = fetch_20newsgroups(subset='train')
        self.newsgroups_test = fetch_20newsgroups(subset='test')

    def get_train(self):
        return (self.newsgroups_train.data, self.newsgroups_train.target)

    def get_test(self):
        return (self.newsgroups_test.data, self.newsgroups_test.target)

    def get_number_classes(self):
        return 20