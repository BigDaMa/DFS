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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import threading
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoMLTreeSpace import MyAutoMLSpace

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





class SpaceGenerator:
    def __init__(self):
        self.classifier_list = get_all_classes(optuna_classifiers)
        self.preprocessor_list = get_all_classes(optuna_preprocessor, addNone=True)
        self.scaling_list = get_all_classes(optuna_scaler, addNone=True)

        self.space = MyAutoMLSpace()

        #generate binary or mapping for each hyperparameter


    def generate_params(self):

        self.space.generate_cat('balanced', [True, False], True)

        category_children = self.space.generate_cat('preprocessor', self.preprocessor_list, IdentityOptuna())
        print(category_children)
        for preprocessor in self.preprocessor_list:
            preprocessor.generate_hyperparameters(self.space)

        self.space.generate_cat('classifier', self.classifier_list, self.classifier_list[0])
        for classifier in self.classifier_list:
            classifier.generate_hyperparameters(self.space)

        self.space.generate_cat('scaler', self.scaling_list, IdentityOptuna())
        for scaler in self.scaling_list:
            scaler.generate_hyperparameters(self.space)

        imputer = SimpleImputerOptuna()
        imputer.generate_hyperparameters(self.space)

        categorical_transformer = OneHotEncoderOptuna()
        scaler.generate_hyperparameters(self.space)


        return self.space

