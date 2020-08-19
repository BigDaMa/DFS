from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoMLProcess import MyAutoML
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

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import threading
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from anytree import Node, RenderTree

from sklearn.ensemble import RandomForestRegressor

from optuna.samplers.random import RandomSampler
import matplotlib.pyplot as plt

#from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

dataset = openml.datasets.get_dataset(31)
#dataset = openml.datasets.get_dataset(1590)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

gen = SpaceGenerator()
space = gen.generate_params()

from anytree import Node, RenderTree

for pre, _, node in RenderTree(space.parameter_tree):
    print("%s%s: %s" % (pre, node.name, node.status))

my_study = optuna.create_study(direction='maximize')

validation_scores = []
test_scores = []

#add Caruana ensemble with replacement # save pipelines to disk

for i in range(1, 10):
    search = MyAutoML(cv=10, number_of_cvs=1, n_jobs=2, time_search_budget=1*60, space=space, study=my_study, main_memory_budget_gb=4)
    best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)
    my_study = search.study

    test_score = auc(search.get_best_pipeline(), X_test, y_test)

    print("budget: " + str(i) + ' => ' + str(best_result) + " test: " + str(test_score))

    validation_scores.append(best_result)
    test_scores.append(test_score)


print(validation_scores)
print(test_scores)

