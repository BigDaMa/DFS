from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import copy

class FeatureTransformations(BaseEstimator, TransformerMixin):

    def transform(self, X, feature_names):
        #add product of global_cv and global_number_cv
        product_cvs = X[:, feature_names.index('global_cv')] * X[:, feature_names.index('global_number_cv')]
        product_hold_out_test = X[:, feature_names.index('hold_out_fraction')] * X[:, feature_names.index('NumberOfInstances')] * X[:, feature_names.index('sample_fraction')]
        product_sampled_data = X[:, feature_names.index('NumberOfInstances')] * X[:, feature_names.index('sample_fraction')]

        number_of_evaluations = X[:, feature_names.index('global_search_time_constraint')] / X[:, feature_names.index('global_evaluation_time_constraint')]

        print('product_cvs: ' + str(product_cvs.shape))
        print('product_hold_out_test: ' + str(product_hold_out_test.shape))
        print('product_sampled_data: ' + str(product_sampled_data.shape))
        print('number_of_evaluations: ' + str(number_of_evaluations.shape))

        #logs 'global_search_time_constraint', 'global_evaluation_time_constraint', 'global_memory_constraint'
        log_global_search_time_constraint = np.log(X[:, feature_names.index('global_search_time_constraint')])
        log_global_evaluation_time_constraint = np.log(X[:, feature_names.index('global_evaluation_time_constraint')])
        log_global_memory_constraint = np.log(X[:, feature_names.index('global_memory_constraint')])


        '''
        return np.hstack((X,
                          product_cvs,
                          product_hold_out_test,
                          product_sampled_data,
                          number_of_evaluations,
                          log_global_search_time_constraint,
                          log_global_evaluation_time_constraint,
                          log_global_memory_constraint))
        '''

    def get_new_feature_names(self, feature_names):
        self.feature_names_new = copy.deepcopy(feature_names)
        self.feature_names_new.append('product_cvs')
        self.feature_names_new.append('hold_out_test_instances')
        self.feature_names_new.append('sampled_instances')
        self.feature_names_new.append('number_of_evaluations')
        self.feature_names_new.append('log_global_search_time_constraint')
        self.feature_names_new.append('log_global_evaluation_time_constraint')
        self.feature_names_new.append('log_global_memory_constraint')
        return self.feature_names_new


    def fit(self, X, y=None):
        return self