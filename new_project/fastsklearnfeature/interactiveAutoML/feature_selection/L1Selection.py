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



class L1Selection(BaseEstimator, SelectorMixin):
    def __init__(self, c_range=[1.0], cv=1):
        name = 'L1Selection'
        self.cv = cv

        model = LogisticRegression(penalty='l1', C=1.0, solver='saga', class_weight='balanced', max_iter=10000, multi_class='auto')
        self.my_pipeline = Pipeline([('scaler', StandardScaler()),
                                ('logreg', model)
                                ])

        self.logregParam = {'logreg__penalty': ['l1'],
                            'logreg__C': c_range,
                            'logreg__solver': ['saga'],
                            'logreg__class_weight': ['balanced'],
                            'logreg__max_iter': [10000],
                            'logreg__multi_class': ['auto']
                            }

    def fit(self, X, y=None):
        if self.cv > 1:
            logregGS = GridSearchCV(estimator=self.my_pipeline, param_grid=self.logregParam, cv=self.cv)
            logregGS.fit(X, y)
            self.feature_mask = logregGS.best_estimator_.named_steps['logreg'].coef_[0] != 0.0
        else:
            self.my_pipeline.fit(X, y)
            self.feature_mask = self.my_pipeline.named_steps['logreg'].coef_[0] != 0.0

        return self

    def _get_support_mask(self):
        return self.feature_mask

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        return True

    def get_sympy_representation(self, input_attributes):
        return None
