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



class MaskSelection(BaseEstimator, SelectorMixin):
    def __init__(self, mask):
        name = 'MaskSelection'
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def _get_support_mask(self):
        return self.mask

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        return True

    def get_sympy_representation(self, input_attributes):
        return None
