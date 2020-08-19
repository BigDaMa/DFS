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

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import threading
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import copy

from anytree import Node, RenderTree

from sklearn.ensemble import RandomForestRegressor

from optuna.samplers.random import RandomSampler
import matplotlib.pyplot as plt
import resource

#size = int(4 * 1024 * 1024 * 1024)
#resource.setrlimit(resource.RLIMIT_AS, (size, resource.RLIM_INFINITY))

#from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

dataset = openml.datasets.get_dataset(31)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)



#print(calculate_all_metafeatures_with_labels(X_train, y, categorical=categorical_indicator, dataset_name='data'))


total_search_time = 2 * 60

def run_AutoML(trial):
    space = None
    search_time = None
    if not 'space' in trial.user_attrs:
        # which hyperparameters to use
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        # which constraints to use
        search_time = trial.suggest_int('global_search_time_constraint', 10, total_search_time, log=False)

        # how much time for each evaluation
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, total_search_time, log=False)

        # how much memory is allowed
        memory_limit = trial.suggest_uniform('global_memory_constraint', 1.5, 4)

        # how many cvs should be used
        cv = trial.suggest_int('global_cv', 2, 20, log=False)

        number_of_cvs = trial.suggest_int('global_number_cv', 1, 10, log=False)

    else:
        space = trial.user_attrs['space']

        print(trial.params)

        #make this a hyperparameter
        search_time = trial.params['global_search_time_constraint']
        evaluation_time = trial.params['global_evaluation_time_constraint']
        memory_limit = trial.params['global_memory_constraint']
        cv = trial.params['global_cv']
        number_of_cvs = trial.params['global_number_cv']


    for pre, _, node in RenderTree(space.parameter_tree):
        print("%s%s: %s" % (pre, node.name, node.status))

    # which dataset to use
    #todo: add more datasets
    dataset = openml.datasets.get_dataset(31)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    search = MyAutoML(cv=cv,
                      number_of_cvs=number_of_cvs,
                      n_jobs=1,
                      evaluation_budget=evaluation_time,
                      time_search_budget=search_time,
                      space=space,
                      main_memory_budget_gb=memory_limit)
    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

    best_pipeline = search.get_best_pipeline()

    test_score = 0.0
    if type(best_pipeline) != type(None):
        test_score = auc(search.get_best_pipeline(), X_test, y_test)


    return test_score


def space2features(space, trial, my_list_constraints_values):
    tuple_param = np.zeros((1, len(my_list)))
    tuple_constraints = np.zeros((1, len(my_list_constraints)))
    t = 0
    for parameter_i in range(len(my_list)):
        tuple_param[t, parameter_i] = space.name2node[my_list[parameter_i]].status


    for constraint_i in range(len(my_list_constraints)):
        tuple_constraints[t, constraint_i] = my_list_constraints_values[constraint_i] #current_trial.params[my_list_constraints[constraint_i]]

    return np.hstack((tuple_param, tuple_constraints))

def predict_range(model, X):
    y_pred = model.predict(X)

    y_pred[y_pred > 1.0] = 1.0
    y_pred[y_pred < 0.0] = 0.0
    return y_pred

def optimize_uncertainty(trial):
    gen = SpaceGenerator()
    space = gen.generate_params()
    space.sample_parameters(trial)

    trial.set_user_attr('space', copy.deepcopy(space))

    search_time = trial.suggest_int('global_search_time_constraint', 10, total_search_time, log=False)
    evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, total_search_time, log=False)
    memory_limit = trial.suggest_uniform('global_memory_constraint', 0.001, 4)
    cv = trial.suggest_int('global_cv', 2, 20, log=False)
    number_of_cvs = trial.suggest_int('global_number_cv', 1, 10, log=False)


    my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]

    features = space2features(space, trial, my_list_constraints_values)

    predictions = []
    for tree in range(model.n_estimators):
        predictions.append(predict_range(model.estimators_[tree], features))

    stddev_pred = np.std(np.matrix(predictions).transpose(), axis=1)

    return stddev_pred[0]

search_time_frozen = 60

def optimize_accuracy_under_constraints(trial):
    gen = SpaceGenerator()
    space = gen.generate_params()
    space.sample_parameters(trial)

    trial.set_user_attr('space', copy.deepcopy(space))

    search_time = trial.suggest_int('global_search_time_constraint', 10, search_time_frozen, log=False)
    evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time_frozen, log=False)
    memory_limit = trial.suggest_uniform('global_memory_constraint', 0.001, 4)
    cv = trial.suggest_int('global_cv', 2, 20, log=False)
    number_of_cvs = trial.suggest_int('global_number_cv', 1, 10, log=False)

    my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]

    features = space2features(space, trial, my_list_constraints_values)

    return predict_range(model, features)


def trial2features(trial):
    X_row_params = np.zeros((1, len(my_list)))
    X_row_constraints = np.zeros((1, len(my_list_constraints)))

    current_trial = trial
    t = 0
    for parameter_i in range(len(my_list)):
        X_row_params[t, parameter_i] = current_trial.user_attrs['space'].name2node[my_list[parameter_i]].status

    for constraint_i in range(len(my_list_constraints)):
        X_row_constraints[t, constraint_i] = current_trial.params[my_list_constraints[constraint_i]]


    X_row_all = np.hstack((X_row_params, X_row_constraints))

    return X_row_all




#random sampling 10 iterations
study = optuna.create_study(direction='maximize', sampler=RandomSampler(seed=42))


#first random sampling
study.optimize(run_AutoML, n_trials=4, n_jobs=1)

print('done')


mgen = SpaceGenerator()
mspace = mgen.generate_params()

my_list = list(mspace.name2node.keys())
my_list.sort()

my_list_constraints = ['global_search_time_constraint', 'global_evaluation_time_constraint', 'global_memory_constraint', 'global_cv', 'global_number_cv']

#generate training data
all_trials = study.get_trials()
X_meta_parameters = np.zeros((len(all_trials), len(my_list)))

X_meta_constraints = np.zeros((len(all_trials), len(my_list_constraints)))
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
        X_meta_parameters[t, parameter_i] = current_trial.user_attrs['space'].name2node[my_list[parameter_i]].status

    for constraint_i in range(len(my_list_constraints)):
        X_meta_constraints[t, constraint_i] = current_trial.params[my_list_constraints[constraint_i]]


X_meta = np.hstack((X_meta_parameters,X_meta_constraints))

feature_names = copy.deepcopy(my_list)
feature_names.extend(my_list_constraints)


pruned_accuray_results = []

while True:

    model = RandomForestRegressor()
    model.fit(X_meta, y_meta)

    #random sampling 10 iterations
    study_uncertainty = optuna.create_study(direction='maximize')
    study_uncertainty.optimize(optimize_uncertainty, n_trials=100, n_jobs=1)

    X_meta = np.vstack((X_meta, trial2features(study_uncertainty.best_trial)))
    y_meta.append(run_AutoML(study_uncertainty.best_trial))

    study_prune = optuna.create_study(direction='maximize')
    study_prune.optimize(optimize_accuracy_under_constraints, n_trials=500, n_jobs=1)

    pruned_accuray_results.append(run_AutoML(study_prune.best_trial))

    plt.plot(range(len(pruned_accuray_results)), pruned_accuray_results)
    plt.show()



    print('Shape: ' + str(X_meta.shape))

    import operator
    def plot_most_important_features(rf_random, names_features, title='importance'):
        importances = {}
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

    plot_most_important_features(model, feature_names)
