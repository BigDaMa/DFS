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
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.CategoricalMissingTransformer import CategoricalMissingTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.OneHotEncoderOptuna import OneHotEncoderOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import threading
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import time
import threading
import resource
import signal
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
        class_list.append(IdentityOptuna())

    return class_list


def signal_handler(sig, frame):
    print('got a signal')
    raise Exception('Signal arrived')

class TimeException(Exception):
    def __init__(self, message="Time is over!"):
        self.message = message
        super().__init__(self.message)


def evaluatePipeline(key,return_dict):
    balanced = mp_global.mp_store[key]['balanced']
    p = mp_global.mp_store[key]['p']
    number_of_cvs = mp_global.mp_store[key]['number_of_cvs']
    cv = mp_global.mp_store[key]['cv']
    scorer = mp_global.mp_store[key]['scorer']
    X = mp_global.mp_store[key]['X']
    y = mp_global.mp_store[key]['y']
    main_memory_budget_gb = mp_global.mp_store[key]['main_memory_budget_gb']

    size = int(main_memory_budget_gb * 1024.0 * 1024.0 * 1024.0)
    resource.setrlimit(resource.RLIMIT_AS, (size, resource.RLIM_INFINITY))

    start_training = time.time()

    if balanced:
        p.fit(X, y, classifier__sample_weight=compute_sample_weight(class_weight='balanced', y=y))
    else:
        p.fit(X, y)
    training_time = time.time() - start_training

    return_dict[key + 'pipeline'] = copy.deepcopy(p)



    scores = []
    for cv_num in range(number_of_cvs):
        my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time())).split(X, y)
        #my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(42)).split(X, y)
        for train_ids, test_ids in my_splits:
            if balanced:
                p.fit(X[train_ids, :], y[train_ids], classifier__sample_weight=compute_sample_weight(class_weight='balanced', y=y[train_ids]))
            else:
                p.fit(X[train_ids, :], y[train_ids])
            scores.append(scorer(p, X[test_ids, :], pd.DataFrame(y[test_ids])))

    return_dict[key + 'result'] = np.mean(scores)





class MyAutoML:
    def __init__(self, cv=5, number_of_cvs=1, evaluation_budget=np.inf, time_search_budget=10*60, n_jobs=1, space=None, study=None, main_memory_budget_gb=4):
        self.cv = cv
        self.time_search_budget = time_search_budget
        self.n_jobs = n_jobs
        self.evaluation_budget = evaluation_budget
        self.number_of_cvs = number_of_cvs

        self.classifier_list = get_all_classes(optuna_classifiers)
        self.preprocessor_list = get_all_classes(optuna_preprocessor, addNone=True)
        self.scaling_list = get_all_classes(optuna_scaler, addNone=True)

        #generate binary or mapping for each hyperparameter


        self.space = space
        self.study = study
        self.main_memory_budget_gb = main_memory_budget_gb

        #print("number of hyperparameters: " + str(len(self.space.parameters_used)))

        #signal.signal(signal.SIGSEGV, signal_handler)


    def get_best_pipeline(self):
        try:
            return self.study.best_trial.user_attrs['pipeline']
        except:
            return None


    def predict(self, X):
        best_pipeline = self.get_best_pipeline()
        return best_pipeline.predict(X)


    def fit(self, X, y, sample_weight=None, categorical_indicator=None, scorer=None):
        self.start_fitting = time.time()

        def objective1(trial):
            start_total = time.time()

            try:

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

                onehot_transformer = OneHotEncoderOptuna()
                onehot_transformer.init_hyperparameters(self.space, X, y)


                numeric_transformer = Pipeline([('imputation', imputer), ('scaler', scaler)])
                categorical_transformer = Pipeline([('removeNAN', CategoricalMissingTransformer()), ('onehot_transform', onehot_transformer)])


                data_preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, np.invert(categorical_indicator)),
                        ('cat', categorical_transformer, categorical_indicator)])

                my_pipeline = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor),
                              ('classifier', classifier)])


                key = 'My_processs' + str(time.time()) + " ## " + str(np.random.randint(0,1000))

                mp_global.mp_store[key] = {}

                mp_global.mp_store[key]['balanced'] = balanced
                mp_global.mp_store[key]['p'] = my_pipeline
                mp_global.mp_store[key]['number_of_cvs'] = self.number_of_cvs
                mp_global.mp_store[key]['cv'] = self.cv
                mp_global.mp_store[key]['scorer'] = scorer
                mp_global.mp_store[key]['X'] = X
                mp_global.mp_store[key]['y'] = y
                mp_global.mp_store[key]['main_memory_budget_gb'] = self.main_memory_budget_gb

                already_used_time = time.time() - self.start_fitting

                if already_used_time + 2 >= self.time_search_budget:  # already over budget
                    time.sleep(2)
                    return -np.inf

                remaining_time = np.min([self.evaluation_budget, self.time_search_budget - already_used_time])

                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                my_process = multiprocessing.Process(target=evaluatePipeline, name='start'+key, args=(key, return_dict,))
                my_process.start()

                my_process.join(int(remaining_time))

                # If thread is active
                while my_process.is_alive():
                    # Terminate foo
                    my_process.terminate()
                    my_process.join()

                del mp_global.mp_store[key]

                result = -np.inf
                if key + 'result' in return_dict:
                    result = return_dict[key + 'result']

                trial.set_user_attr('total_time', time.time() - start_total)

                try:
                    if self.study.best_value < result:
                        trial.set_user_attr('pipeline', return_dict[key + 'pipeline'])
                except:
                    trial.set_user_attr('pipeline', return_dict[key + 'pipeline'])

                return result
            except:
                return -np.inf

        if type(self.study) == type(None):
            self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective1, timeout=self.time_search_budget, n_jobs=self.n_jobs, catch=(TimeException,))
        return self.study.best_value




if __name__ == "__main__":
    auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

    dataset = openml.datasets.get_dataset(1590)

    #dataset = openml.datasets.get_dataset(31)
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

    search = MyAutoML(cv=5, n_jobs=2, time_search_budget=240, space=space)

    begin = time.time()

    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

    print(time.time() - begin)