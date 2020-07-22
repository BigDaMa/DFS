from abc import ABCMeta, abstractmethod
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.private_models.Smooth_Random_Trees import DP_Random_Forest
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
import pickle
import copy

class PrivateDecisionTree(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def fit(self, X, y, sample_weight=None):

        self.n_outputs_ = y.shape[1]
        self.n_features_ = X.shape[1]

        new_y = copy.deepcopy(y)
        if isinstance(y, pd.DataFrame):
            new_y = new_y.values
        new_y = new_y.reshape(-1,1)

        self.classes_ = np.unique(new_y)

        all_data = np.hstack((new_y,X))

        self.model = DP_Random_Forest(all_data, epsilon=self.epsilon)
        return self

    def predict(self, X):
        empty = np.zeros((len(X), 1))
        all_data = np.hstack((empty, X))
        return np.asarray(self.model.predict(all_data))

''' A toy example of how to call the class '''
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    diabetes = load_iris()

    X = diabetes.data
    y = diabetes.target

    X = pickle.load(open('/tmp/X.pkl', 'rb'))
    y = pickle.load(open('/tmp/y.pkl', 'rb'))


    print(y)

    model = PrivateDecisionTree(epsilon=0.01)
    model.fit(X, y)



    print(model.predict(X))

    import numpy as np
    from art.classifiers import SklearnClassifier

    import copy
    from art.attacks.evasion import HopSkipJump

    classifier = SklearnClassifier(model=model)
    attack = HopSkipJump(classifier=classifier, max_iter=1, max_eval=100)

    X_test_adv = attack.generate(X)

    print(model.predict(X_test_adv))