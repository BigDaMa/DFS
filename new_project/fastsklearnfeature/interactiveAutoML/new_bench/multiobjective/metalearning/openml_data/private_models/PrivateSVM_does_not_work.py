from abc import ABCMeta, abstractmethod
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.private_models.Smooth_Random_Trees import DP_Random_Forest
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import copy
import numpy as np
#from statistics import stdev
# from pylab import norm
from scipy.optimize import minimize

'''
code from https://gitlab.com/dp-stats/dp-stats/-/tree/master/
'''

def noisevector( scale, Length ):

    r1 = np.random.normal(0, 1, Length)#standard normal distribution
    n1 = np.linalg.norm( r1, 2 )#get the norm of this random vector
    r2 = r1 / n1#the norm of r2 is 1
    normn = np.random.gamma( Length, 1/scale, 1 )#Generate the norm of noise according to gamma distribution
    res = r2 * normn#get the result noise vector
    return res

def huber(z, h):#chaudhuri2011differentially corollary 21

    if z > 1 + h:
        hb = 0
    elif np.fabs(1-z) <= h:
        hb = (1 + h - z)**2 / (4 * h)
    else:
        hb = 1 - z
    return hb

def svm_output_train(data, labels, epsilon, Lambda, h):

    N = len( labels )
    l = len( data[0] )#length of a data point
    scale = N * Lambda * epsilon / 2
    noise = noisevector( scale, l )
    x0 = np.zeros(l)#starting point with same length as any data point

    def obj_func(x):
        jfd = huber( labels[0] * np.dot(data[0],x), h )
        for i in range(1,N):
            jfd = jfd + huber( labels[i] * np.dot(data[i],x), h )
        f = ( 1/N )*jfd + (1/2) * Lambda * ( np.linalg.norm(x,2)**2 )
        return f

    #minimization procedure
    f = minimize(obj_func, x0, method='Nelder-Mead').x #empirical risk minimization using scipy.optimize minimize function
    fpriv = f + noise
    return fpriv

def svm_objective_train(data, labels,  epsilon, Lambda, h):

    #parameters in objective perturbation method
    c = 1 / ( 2 * h )#chaudhuri2011differentially corollary 13
    N = len( labels )#number of data points in the data set
    l = len( data[0] )#length of a data point
    x0 = np.zeros(l)#starting point with same length as any data point
    Epsilonp = epsilon - 2 * np.log( 1 + c / (Lambda * N))
    if Epsilonp > 0:
        Delta = 0
    else:
        Delta = c / ( N * (np.exp(epsilon/4)-1) ) - Lambda
        Epsilonp = epsilon / 2
    noise = noisevector(Epsilonp/2, l)

    def obj_func(x):
        jfd = huber( labels[0] * np.dot(data[0], x), h)
        for i in range(1,N):
            jfd = jfd + huber( labels[i] * np.dot(data[i], x), h )
        f = (1/N) * jfd + (1/2) * Lambda * (np.linalg.norm(x,2)**2) + (1/N) * np.dot(noise,x) + (1/2)*Delta*(np.linalg.norm(x,2)**2)
        return f

    #minimization procedure
    fpriv = minimize(obj_func, x0, method='Nelder-Mead').x#empirical risk minimization using scipy.optimize minimize function
    return fpriv

import pandas as pd

class PrivateSVM(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, method='obj', epsilon=0.1, Lambda = 0.01, h = 0.5):
        self.epsilon = epsilon
        self.Lambda = Lambda
        self.h = h
        self.method = method

    def fit(self, X, y, sample_weight=None):

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        new_y = copy.deepcopy(y)
        if isinstance(y, pd.DataFrame):
            new_y = copy.deepcopy(new_y.values)

        self.classes_ = []
        self.n_classes_ = []

        self.n_outputs_ = new_y.shape[1]

        self.classes_ = np.unique(new_y)

        if self.method == 'obj':
            self.fpriv = svm_objective_train(X, new_y, self.epsilon, self.Lambda, self.h)
        else:
            self.fpriv = svm_output_train(X, new_y, self.epsilon, self.Lambda, self.h)
        return self

    def predict(self, X):
        indices = (np.dot(X, self.fpriv) > 0).astype(np.int)
        return np.asarray(self.classes_[indices], dtype=np.intp)


''' A toy example of how to call the class '''
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import scale
    from sklearn.metrics import f1_score
    diabetes = load_wine()

    X = diabetes.data
    y = diabetes.target >0

    #X = scale(X)

    print(y)

    model = PrivateSVM(epsilon=10000)
    model.fit(X, pd.DataFrame(y))
    predictions = model.predict(X)

    print(predictions)
    print(predictions.shape)

    print(f1_score(y, predictions))


    import numpy as np
    from art.classifiers import SklearnClassifier

    import copy
    from art.attacks.evasion import HopSkipJump

    classifier = SklearnClassifier(model=model)
    attack = HopSkipJump(classifier=classifier, max_iter=1, max_eval=100)

    X_test_adv = attack.generate(X)


