from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class NaiveBayes():
    def __init__(self):
        self.parameter_search_space = \
            [{'alpha': np.logspace(-3, 3, 7)}]
        self.model = None
        self.classes = []

    def get_classes(self):
        return self.model.classes_


    def optimize_hyperparameters(self, x, y, folds, jobs=4):
        clf = GridSearchCV(MultinomialNB(), self.parameter_search_space, cv=folds, n_jobs=jobs)
        clf.fit(x, y)
        self.best_params = clf.best_params_


    def train(self, x, y, params=None):
        parameters = params
        if params == None:
            parameters = self.best_params
        self.model = MultinomialNB(**parameters)
        self.model.fit(x, y)

    def partial_train(self, x, y, params=None):
        parameters = params
        if params == None:
            parameters = self.best_params

        if len(self.classes) == 0:
            self.classes = np.unique(y)
            self.model = MultinomialNB(**parameters)

        self.model.partial_fit(x, y, classes=self.classes)


    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[0]