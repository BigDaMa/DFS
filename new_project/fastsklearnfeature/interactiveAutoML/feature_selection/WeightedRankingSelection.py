from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sympy
import numpy as np
from sklearn.model_selection import GridSearchCV

import copy

class WeightedRankingSelection(BaseEstimator, SelectorMixin):
    def __init__(self, scores, weights, k, names=None):
        self.scores = scores
        self.weights = weights
        self.k = k
        self.names = names


    def fit(self, X, y=None):
        for score_i in range(len(self.scores)):
            self.scores[score_i] /= np.sum(self.scores[score_i])
            self.scores[score_i] *= self.weights[score_i]

        final_scores = np.sum(np.matrix(self.scores), axis=0).A1
        ids = np.argsort(final_scores *-1)
        self.feature_mask = np.zeros(len(self.scores[0]), dtype=bool)
        self.feature_mask[ids[0:self.k]] = True


        #print('hallo: ' + str(ids[0:self.k]))
        print('seledcted features: ' + str(self.names[ids[0:self.k]]))

        return self

    def _get_support_mask(self):
        return self.feature_mask
