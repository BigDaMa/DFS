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
    def __init__(self, scores, weights, k, names=None, hyperparameter_mask=None):
        self.scores = scores
        self.weights = weights
        self.k = k
        self.names = names
        self.hyperparameter_mask = hyperparameter_mask

    def fit(self, X, y=None):
        agg_scores = copy.deepcopy(self.scores)
        for score_i in range(len(agg_scores)):
            agg_scores[score_i] = MinMaxScaler().fit_transform(np.array(agg_scores[score_i]).reshape(-1,1)).flatten() #values between 0 and 1
            agg_scores[score_i] /= np.sum(agg_scores[score_i]) #get unit norm
            agg_scores[score_i] *= self.weights[score_i]

        final_scores = np.sum(np.matrix(agg_scores), axis=0).A1
        ids = np.argsort(final_scores *-1)
        self.feature_mask = np.zeros(len(self.scores[0]), dtype=bool)
        self.feature_mask[ids[0:self.k]] = True

        if type(self.hyperparameter_mask) != type(None):
            for feature_i in range(len(self.hyperparameter_mask)):
                if self.hyperparameter_mask[feature_i] == False:
                    self.feature_mask[feature_i] = False

        return self

    def _get_support_mask(self):
        return self.feature_mask
