from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class SVC_Model():
    def __init__(self):
        self.parameter_search_space = \
            [{'kernel': ['rbf'],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
              'probability': [True]},
             {'kernel': ['linear'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
              'probability': [True]}]
        self.model = None

    def get_classes(self):
        return self.model.classes_


    def optimize_hyperparameters(self, x, y, folds, jobs=4):
        clf = GridSearchCV(SVC(), self.parameter_search_space, cv=folds, n_jobs=jobs)
        clf.fit(x, y)
        self.best_params = clf.best_params_


    def train(self, x, y, params=None):
        parameters = params
        if params == None:
            parameters = self.best_params
        self.model = SVC(**parameters)
        self.model.fit(x, y)


    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[0]