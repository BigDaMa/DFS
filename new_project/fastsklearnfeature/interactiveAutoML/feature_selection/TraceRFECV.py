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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from sklearn.feature_selection import RFECV


class TraceRFECV(BaseEstimator, SelectorMixin):
    def __init__(self, max_complexity=None, min_accuracy=None, model=None, parameters=None, cv=None, scoring=None):
        self.parameters = parameters
        self.cv = cv
        self.max_complexity = max_complexity
        self.min_accuracy = min_accuracy
        self.scoring = scoring
        self.model = model
        self.step_size = 1

    def fit(self, X, y=None):
        self.scores_ = []
        self.mask_ = np.ones(X.shape[1], dtype=bool)
        ids = list(range(X.shape[1]))

        for i in range(X.shape[1] - self.step_size, 0, -self.step_size):
            selector = RFECV(self.model, min_features_to_select=i, step=self.step_size, cv=self.cv, scoring=self.scoring)
            selector.fit(X[:, ids], y)
            self.scores_.append(selector.grid_scores_[0])
            print("ids: " + str(len(ids)))
            print("support: " + str(len(selector.support_)))
            print(selector.grid_scores_)

            for d in range(len(selector.support_)):
                if not selector.support_[d]:
                    self.mask_[ids[d]] = False
                    del ids[d]
            if self.max_complexity >= i and self.scores_[-1] >= self.min_accuracy:
                return self




    def _get_support_mask(self):
        return self.mask_