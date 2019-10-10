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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


class RedundancyRemoval(BaseEstimator, SelectorMixin):
    def __init__(self, regressor=KNeighborsRegressor(), r2_threshold=0.0, cv=10):
        self.regressor = regressor
        self.r2_threshold = r2_threshold
        self.cv = cv

    def fit(self, X, y=None):
        self.feature_mask_ = np.zeros(X.shape[1], dtype=bool)

        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('logreg', self.regressor)
                                ])

        #ToDO: remove deselected features from regression

        for col_i in range(X.shape[1]):
            r2_score = np.mean(cross_val_score(pipeline, X[:, [x for x in range(X.shape[1]) if x != col_i]], X[:, col_i], cv=self.cv, scoring = 'r2'))
            print(col_i)
            if r2_score > self.r2_threshold:
                self.feature_mask_[col_i] = True

        return self

    def _get_support_mask(self):
        return self.feature_mask_