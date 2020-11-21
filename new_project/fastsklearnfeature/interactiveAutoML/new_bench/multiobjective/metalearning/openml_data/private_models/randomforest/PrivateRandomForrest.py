from abc import ABCMeta, abstractmethod
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.private_models.randomforest.samfletcherforest import DP_Random_Forest
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import copy
import numpy as np

class PrivateRandomForest(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, n_estimators, epsilon, max_depth=None):
        self.n_estimators = n_estimators
        self.epsilon = epsilon
        self.max_depth = max_depth

    def fit(self, X, y, sample_weight=None):

        self.n_outputs_ = 1
        self.n_features_ = X.shape[1]

        cats = []
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) <= 2:
                cats.append(i + 1)

        new_y = copy.deepcopy(y)
        if isinstance(y, pd.DataFrame):
            new_y = new_y.values
        new_y = new_y.reshape(-1,1)

        self.classes_ = np.unique(new_y)

        all_data = np.hstack((new_y,X))

        self.model = DP_Random_Forest(all_data, epsilon=self.epsilon, num_trees=self.n_estimators, categs=cats, max_depth=self.max_depth)
        return self

    def predict(self, X):
        empty = np.zeros((len(X), 1))
        all_data = np.hstack((empty, X))
        return np.asarray(self.model.predict(all_data))

# A toy example of how to call the class
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import f1_score
    diabetes = load_breast_cancer()

    X = diabetes.data
    y = diabetes.target

    model = PrivateRandomForest(n_estimators=100, epsilon=0.1)
    model.fit(X, y)

    print(f1_score(y,model.predict(X)))




    #print(model.predict(X))

    import numpy as np
    from art.classifiers import SklearnClassifier

    import copy
    from art.attacks.evasion import HopSkipJump

    classifier = SklearnClassifier(model=model)
    attack = HopSkipJump(classifier=classifier, max_iter=1, max_eval=100)

    X_test_adv = attack.generate(X)

    print(model.predict(X_test_adv))
