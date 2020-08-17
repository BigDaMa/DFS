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
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import threading
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import time
import threading

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





class MyAutoML:
    def __init__(self, cv=5, evaluation_budget=np.inf, time_search_budget=10*60, n_jobs=1, space=None):
        self.cv = cv
        self.time_search_budget = time_search_budget
        self.n_jobs = n_jobs
        self.evaluation_budget = evaluation_budget

        self.classifier_list = get_all_classes(optuna_classifiers)
        self.preprocessor_list = get_all_classes(optuna_preprocessor, addNone=True)
        self.scaling_list = get_all_classes(optuna_scaler, addNone=True)

        #generate binary or mapping for each hyperparameter


        self.space = space

        #print("number of hyperparameters: " + str(len(self.space.parameters_used)))



    def fit(self, X, y, sample_weight=None, categorical_indicator=None, scorer=None):
        self.start_fitting = time.time()

        def objective1(trial, return_dict):
            start_total = time.time()

            self.space.trial = trial

            preprocessor = self.space.suggest_categorical('preprocessor', self.preprocessor_list)
            preprocessor.init_hyperparameters(self.space, X, y)

            classifier = self.space.suggest_categorical('classifier', self.classifier_list)
            classifier.init_hyperparameters(self.space, X, y)

            balanced = False
            if isinstance(classifier, KNeighborsClassifierOptuna) or \
                    isinstance(classifier, QuadraticDiscriminantAnalysisOptuna) or \
                    isinstance(classifier, PassiveAggressiveOptuna) or \
                    isinstance(classifier, HistGradientBoostingClassifierOptuna):
                balanced = False
            else:
                balanced = self.space.suggest_categorical('balanced', [True, False])

            imputer = SimpleImputerOptuna()
            imputer.init_hyperparameters(self.space, X, y)

            scaler = self.space.suggest_categorical('scaler', self.scaling_list)
            scaler.init_hyperparameters(self.space, X, y)

            categorical_transformer = OneHotEncoderOptuna()
            scaler.init_hyperparameters(self.space, X, y)

            try:
                numeric_transformer = Pipeline([('imputation', imputer), ('scaler', scaler)])

                data_preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, np.invert(categorical_indicator)),
                        ('cat', categorical_transformer, categorical_indicator)])

                p = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor),
                              ('classifier', classifier)])

                start_training = time.time()

                if balanced:
                    p.fit(X, y,
                          classifier__sample_weight=compute_sample_weight(class_weight='balanced', y=y))
                else:
                    p.fit(X, y)
                training_time = time.time() - start_training
                trial.set_user_attr('training_time', training_time)

                scores = []
                my_splits = StratifiedKFold(n_splits=self.cv).split(X, y)
                for train_ids, test_ids in my_splits:
                    if balanced:
                        p.fit(X[train_ids, :], y[train_ids], classifier__sample_weight=compute_sample_weight(class_weight='balanced', y=y[train_ids]))
                    else:
                        p.fit(X[train_ids, :], y[train_ids])
                    scores.append(scorer(p, X[test_ids, :], pd.DataFrame(y[test_ids])))

                trial.set_user_attr('total_time', time.time() - start_total)

                return_dict['value'] = np.mean(scores)
            except Exception as e:
                print(p)
                print(str(e))
                try:
                    trial.set_user_attr('total_time', time.time() - start_total)
                    return_dict['value'] = -np.inf
                except:
                    pass

        def objective(trial):

            already_used_time = time.time() - self.start_fitting

            if already_used_time >= self.time_search_budget: #already over budget
                return -np.inf

            remaining_time = np.min([self.evaluation_budget, self.time_search_budget - already_used_time])

            return_dict = {}
            return_dict['value'] = -np.inf
            # Start foo as a process

            t = threading.Thread(target=objective1, args=(trial, return_dict))
            t.daemon = True
            t.start()

            t.join(remaining_time)

            return return_dict['value']

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, timeout=self.time_search_budget, n_jobs=self.n_jobs)
        return study.best_value

if __name__ == "__main__":
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

    search = MyAutoML(cv=5, n_jobs=6, time_search_budget=120, space=space)

    begin = time.time()

    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

    print(time.time() - begin)