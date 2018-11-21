from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier:
    def __init__(self, number_classes, score):
        self.number_classes = number_classes
        self.score = score


    def fit(self, train, test, best_params=None):
        if type(best_params) == type(None):
            best_params = {}
        model = LogisticRegression(**best_params)
        model.fit(train, test)
        return model


    def get_classifier(self):
        return LogisticRegression()


    def run_cross_validation(self, train, train_target, folds):
        cv_params = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        scorer = self.score.get_scorer()
        optimized_GBM = GridSearchCV(LogisticRegression(),
                                     cv_params,
                                     scoring=scorer, cv=folds, n_jobs=1, verbose=0)

        optimized_GBM.fit(train, train_target)

        our_params = {}
        our_params.update(optimized_GBM.best_params_)

        return our_params
