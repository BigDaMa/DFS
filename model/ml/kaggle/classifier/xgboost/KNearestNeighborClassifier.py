from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighborClassifier:
    def __init__(self, number_classes, score):
        self.number_classes = number_classes
        self.score = score


    def fit(self, train, test, best_params=None):
        if type(best_params) == type(None):
            best_params = {}
        model = KNeighborsClassifier(**best_params)
        model.fit(train, test)
        return model


    def run_cross_validation(self, train, train_target, folds):
        cv_params = {"n_neighbors": list(range(1,15))}
        scorer = self.score.get_scorer()
        optimized_GBM = GridSearchCV(KNeighborsClassifier(),
                                     cv_params,
                                     scoring=scorer, cv=folds, n_jobs=1, verbose=0)

        optimized_GBM.fit(train, train_target)

        our_params = {}
        our_params.update(optimized_GBM.best_params_)

        return our_params
