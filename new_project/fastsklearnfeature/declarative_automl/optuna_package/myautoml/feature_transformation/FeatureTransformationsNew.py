from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import copy

class FeatureTransformations(BaseEstimator, TransformerMixin):

    def transform(self, X, feature_names):

        #logs
        log_global_search_time_constraint = np.log(X[:, feature_names.index('global_search_time_constraint')])
        log_global_memory_constraint = np.log(X[:, feature_names.index('global_memory_constraint')])
        log_privacy = np.log(X[:, feature_names.index('privacy')])

        has_privacy_constraint = X[:, feature_names.index('privacy')] < 1000


        return np.hstack((X,
                          log_global_search_time_constraint.reshape((1, 1)),
                          log_global_memory_constraint.reshape((1, 1)),
                          log_privacy.reshape((1, 1)),
                          has_privacy_constraint.reshape((1, 1))
                        ))


    def get_new_feature_names(self, feature_names):
        self.feature_names_new = copy.deepcopy(feature_names)
        self.feature_names_new.append('log_search_time_constraint')
        self.feature_names_new.append('log_search_memory_constraint')
        self.feature_names_new.append('log_privacy_constraint')
        self.feature_names_new.append('has_privacy_constraint')
        return self.feature_names_new


    def fit(self, X, y=None):
        return self