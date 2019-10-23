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
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import scipy.special
import time


class ALSelectionK(BaseEstimator, SelectorMixin):
    def __init__(self, model=None, parameters=None, kfold=None, scoring=None, batch_size=1, sample_size=10000, max_complexity=None, min_accuracy=None, fit_time_out=None):
        name = 'L1Selection'
        self.kfold = kfold
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.model = model
        self.parameters = parameters
        self.score = scoring
        self.numberTrees = 10
        self.max_complexity = max_complexity
        self.fit_time_out = fit_time_out
        self.min_accuracy = min_accuracy

    def fit(self, X, y=None):
        start_time = time.time()

        def predict_range(model, X):
            y_pred = model.predict(X)

            y_pred[y_pred > 1.0] = 1.0
            y_pred[y_pred < 0.0] = 0.0
            return y_pred

        def calculate_average_score_per_feature(X_train, y_train):
            feature_count = np.count_nonzero(X_train, axis=0)
            sum_per_feature = np.matrix(y_train) * X_train

            average_per_feature = np.divide(sum_per_feature, feature_count, out=np.zeros_like(sum_per_feature), where=feature_count != 0)
            return average_per_feature.A1

        #first run for all single features
        p = np.zeros(X.shape[1])
        X_train = np.zeros((X.shape[1], X.shape[1]), dtype=bool)
        for col_i in range(X.shape[1]):
            logregGS = GridSearchCV(estimator=self.model, param_grid=self.parameters, cv=self.kfold, scoring=self.score)
            logregGS.fit(X[:, col_i].reshape(-1, 1), y)
            p[col_i] = logregGS.best_score_
            print(str(col_i) + ': ' + str(logregGS.best_score_))
            X_train[col_i, col_i] = True

        y_train = p

        #probabilities = calculate_average_score_per_feature(X_train, y_train)
        #probabilities = probabilities / np.sum(probabilities)

        number_of_features = self.max_complexity

        while(True):
            #train model to estimate accuracy
            model = RandomForestRegressor(n_estimators=self.numberTrees)
            model.fit(X_train, y_train)
            print('model fitted')

            # https://stackoverflow.com/questions/57408148/numpy-random-choice-with-probabilities-to-produce-a-2d-array-with-unique-rows
            sample_size = int(scipy.special.comb(X.shape[1], number_of_features))
            if self.sample_size < sample_size:
                sample_size = self.sample_size

            print("sample size: " + str(sample_size))

            # sample pairs
            # https://stackoverflow.com/questions/50765131/how-to-efficiently-convert-a-list-into-probability-distribution

            #average score per feature
            probabilities = calculate_average_score_per_feature(X_train, y_train)
            probabilities = probabilities / np.sum(probabilities)

            my_combinations = set()
            while len(my_combinations) < sample_size:
                new_combination = np.random.choice(X.shape[1], size=number_of_features, replace=False, p=probabilities)
                #new_combination = np.random.choice(X.shape[1], size=10, replace=False, p=probabilities)
                my_combinations.add(frozenset(new_combination))

            #print(my_combinations)

            X_test_pairs = np.zeros((sample_size, X.shape[1]), dtype=bool)
            my_combinations_list = list(my_combinations)

            for combo_i in range(len(my_combinations_list)):
                for element in my_combinations_list[combo_i]:
                    X_test_pairs[combo_i, element] = True

            # estimate accuracy of sampled pairs
            estimated_accuracy = predict_range(model, X_test_pairs)
            accuracy_sorted_ids = np.argsort(estimated_accuracy * -1)

            # calculate uncertainty of predictions for sampled pairs
            predictions = []
            for tree in range(self.numberTrees):
                predictions.append(predict_range(model.estimators_[tree], X_test_pairs))

            uncertainty = np.matrix(np.std(np.matrix(predictions).transpose(), axis=1)).A1

            print('mean uncertainty: ' + str(np.average(uncertainty)))

            #get top k pairs with respect accuracy and uncertainty
            uncertainty_sorted_ids = np.argsort(uncertainty * -1)

            sorted = set(uncertainty_sorted_ids[0:int(self.batch_size)])


            errors = []
            for sorted_id in sorted:
                logregGS = GridSearchCV(estimator=self.model, param_grid=self.parameters, cv=self.kfold, scoring=self.score)
                logregGS.fit(X[:, list(my_combinations_list[sorted_id])], y)
                print('shape: ' + str(X[:, list(my_combinations_list[sorted_id])].shape))
                y_train = np.append(y_train, logregGS.best_score_)
                X_train = np.vstack([X_train, X_test_pairs[sorted_id]])

                error = np.abs(estimated_accuracy[sorted_id] - logregGS.best_score_)
                print('error: ' + str(error))
                errors.append(error)
            print("average error: " + str(np.mean(errors)))

            best_ids = np.argsort(y_train*-1)
            if y_train[best_ids[0]] >= self.min_accuracy:
                print("acc: " + str(y_train[best_ids[0]]))
                self.feature_mask = X_train[best_ids[0], :]
                return self

            if type(self.fit_time_out) != type(None) and self.fit_time_out < time.time() - start_time:
                return self

    def _get_support_mask(self):
        return self.feature_mask

    def is_applicable(self, feature_combination: List[CandidateFeature]):
        return True

    def get_sympy_representation(self, input_attributes):
        return None
