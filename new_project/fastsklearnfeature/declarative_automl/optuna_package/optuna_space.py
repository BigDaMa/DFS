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
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import categorical
from fastsklearnfeature.declarative_automl.optuna_package.IdentityOptuna import IdentityOptuna
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.OneHotEncoderOptuna import OneHotEncoderOptuna

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_all_classes(my_module, addNone=False):
    clsmembers = inspect.getmembers(my_module, inspect.ismodule)
    class_list = []
    for member in clsmembers:
        member_classes = inspect.getmembers(member[1])
        for mclass in member_classes:
            if 'Optuna' in mclass[0]:
                class_list.append(mclass[1]())

    if addNone:
        class_list.append(IdentityOptuna())

    return class_list

classifier_list = get_all_classes(optuna_classifiers)
preprocessor_list = get_all_classes(optuna_preprocessor, addNone=True)
scaling_list = get_all_classes(optuna_preprocessor, addNone=True)


auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

dataset = openml.datasets.get_dataset(31)
#dataset = openml.datasets.get_dataset(1590)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

print(preprocessor_list)

def objective(trial):
    start_total = time.time()

    preprocessor = categorical(trial, 'preprocessor', preprocessor_list)
    preprocessor.init_hyperparameters(trial, X_train, y_train)

    classifier = trial.suggest_categorical('classifier', classifier_list)
    classifier.init_hyperparameters(trial, X_train, y_train)

    balanced = False
    if not (isinstance(classifier, KNeighborsClassifier) or
            isinstance(classifier, QuadraticDiscriminantAnalysis) or
            isinstance(classifier, PassiveAggressiveClassifier)
    ):
        balanced = trial.suggest_categorical('balanced', [True, False])

    imputer = SimpleImputerOptuna()
    imputer.init_hyperparameters(trial, X_train, y_train)

    scaler = trial.suggest_categorical('scaler', scaling_list)
    scaler.init_hyperparameters(trial, X_train, y_train)

    categorical_transformer = OneHotEncoderOptuna()
    scaler.init_hyperparameters(trial, X_train, y_train)

    try:
        numeric_transformer = Pipeline([('imputation', imputer), ('scaler', scaler)])

        data_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, np.invert(categorical_indicator)),
                ('cat', categorical_transformer, categorical_indicator)])

        p = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor), ('classifier', classifier)])

        start_training = time.time()

        if balanced:
            p.fit(X_train, y_train, classifier__sample_weight=compute_sample_weight(class_weight='balanced', y=y_train))
        else:
            p.fit(X_train, y_train)
        training_time = time.time() - start_training
        trial.set_user_attr('training_time', training_time)

        #todo implement cv to use balancing in cv
        scores = cross_val_score(p, X_train, y_train, cv=5, scoring=auc)

        trial.set_user_attr('total_time', time.time() - start_total)

        return np.mean(scores)
    except Exception as e:
        print(p)
        print(str(e))
        trial.set_user_attr('total_time', time.time() - start_total)
        return -np.inf

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000, n_jobs=8)

print(study.best_trial)

pickle.dump(study, open("/tmp/optuna_study.p", "wb"))