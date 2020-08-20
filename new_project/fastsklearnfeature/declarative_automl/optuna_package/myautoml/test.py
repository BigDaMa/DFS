import optuna
from sklearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
import numpy as np
import inspect
import fastsklearnfeature.declarative_automl.optuna_package.classifiers as optuna_classifiers
import fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing as optuna_preprocessor
import fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling as optuna_scaler
import fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.categorical_encoding as optuna_categorical_encoding

from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.MyIdentity import IdentityTransformation
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.CategoricalMissingTransformer import CategoricalMissingTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import time
import resource
import copy

import fastsklearnfeature.declarative_automl.optuna_package.myautoml.automl_parameters as mp_global

import multiprocessing

def get_all_classes(my_module, addNone=False):
    clsmembers = inspect.getmembers(my_module, inspect.ismodule)
    class_list = []
    for member in clsmembers:
        member_classes = inspect.getmembers(member[1])
        for mclass in member_classes:
            if 'Optuna' in mclass[0]:
                class_list.append(mclass[1]())

    if addNone:
        class_list.append(IdentityTransformation())

    return class_list



print(get_all_classes(optuna_classifiers))