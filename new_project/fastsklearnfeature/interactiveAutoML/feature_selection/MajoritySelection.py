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



class MajoritySelection(BaseEstimator, SelectorMixin):
    def __init__(self, selection_strategies):
        self.selection_strategies = selection_strategies

    def fit(self, X, y=None):
        for s_i in range(len(self.selection_strategies)):
            self.selection_strategies[s_i].fit(X, y)
        return self

    def _get_support_mask(self):
        majority_mask = self.selection_strategies[0]._get_support_mask()
        for s in self.selection_strategies:
            majority_mask = np.logical_and(majority_mask, s._get_support_mask())
        return majority_mask

