from abc import ABCMeta, abstractmethod
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.private_models.Smooth_Random_Trees import DP_Random_Forest
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class PrivateDecisionTree(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, epsilon):
        self.epsilon = epsilon


    def fit(self, X, y, sample_weight=None):
        all_data = np.hstack((y.reshape(-1,1),X))
        print(all_data.shape)
        self.model = DP_Random_Forest(all_data, epsilon=self.epsilon)
        return self

    def predict(self, X):
        empty = np.zeros((len(X),1))
        all_data = np.hstack((empty, X))
        return self.model.predict(all_data)

''' A toy example of how to call the class '''
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    diabetes = load_iris()

    X = diabetes.data
    y = diabetes.target

    print(y)

    model = PrivateDecisionTree(epsilon=0.01)
    model.fit(X,y)
    print(model.predict(X))