from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoML import MyAutoML
import optuna
from sklearn.pipeline import Pipeline
import pickle
import time
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import sys, inspect
from fastsklearnfeature.declarative_automl.optuna_package.classifiers import *
import fastsklearnfeature.declarative_automl.optuna_package.classifiers as optuna_classifiers
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing import *
import fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing as optuna_preprocessor
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling import *
import fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling as optuna_scaler


from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import categorical
from fastsklearnfeature.declarative_automl.optuna_package.IdentityOptuna import IdentityOptuna
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.OneHotEncoderOptuna import OneHotEncoderOptuna

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_Generation import SpaceGenerator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import threading
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from optuna.samplers.random import RandomSampler
import matplotlib.pyplot as plt

from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

dataset = openml.datasets.get_dataset(31)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)



print(calculate_all_metafeatures_with_labels(X_train, y, categorical=categorical_indicator, dataset_name='data'))




def run_AutoML(trial):
    # which hyperparameters to use
    gen = SpaceGenerator()
    space = gen.generate_params()
    space.sample_parameters(trial)

    # which dataset to use
    #todo: add more datasets
    dataset = openml.datasets.get_dataset(31)

    # which constraints to use
    #search_time = trial.suggest_int('global_search_time_constraint', 10, 10 * 60, log=False)
    search_time = trial.suggest_int('global_search_time_constraint', 10, 2 * 60, log=False)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    search = MyAutoML(cv=5, n_jobs=1, time_limit=search_time, space=space)
    best_value = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)
    return best_value



#random sampling 10 iterations
study = optuna.create_study(direction='maximize', sampler=RandomSampler(seed=42))

while True:
    study.optimize(run_AutoML, n_trials=2, n_jobs=1)

    print('done')

    my_list = list(study.get_trials()[0].params.keys())
    my_list.sort()

    #generate training data
    all_trials = study.get_trials()
    X_meta = np.zeros((len(all_trials), len(all_trials[0].params)))
    y_meta = []

    #todo: create metafeatures for dataset
    #todo: add log scaled search time

    for t in range(len(study.get_trials())):
        current_trial = all_trials[t]
        if current_trial.value >= 0 and current_trial.value <= 1.0:
            y_meta.append(current_trial.value)
        else:
            y_meta.append(0.0)
        for parameter_i in range(len(my_list)):
            X_meta[t, parameter_i] = current_trial.params[my_list[parameter_i]]


    model = RandomForestRegressor()
    model.fit(X_meta, y_meta)

    print('Shape: ' + str(X_meta.shape))

    import operator
    def plot_most_important_features(rf_random, names_features, title='importance'):
        importances =  {}
        for name_i in range(len(names_features)):
            importances[names_features[name_i]] = rf_random.feature_importances_[name_i]

        sorted_x = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)

        labels = []
        score = []
        t = 0
        for key, value in sorted_x:
            labels.append(key)
            score.append(value)
            t += 1
            if t == 25:
                break

        ind = np.arange(len(score))
        plt.bar(ind, score, align='center', alpha=0.5)
        #plt.yticks(ind, labels)

        plt.xticks(ind, labels, rotation='vertical')

        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.6)
        plt.show()

    plot_most_important_features(model, my_list)

#generate training data


#uncertainty sampling