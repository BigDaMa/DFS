import pickle
from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoMLProcess import MyAutoML
import optuna
import time
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator
import copy
from optuna.trial import FrozenTrial
from anytree import RenderTree
from sklearn.ensemble import RandomForestRegressor
from optuna.samplers.random import RandomSampler
import matplotlib.pyplot as plt
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import optimize_accuracy_under_constraints
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import run_AutoML
from anytree import RenderTree

metafeature_names_new = ['ClassEntropy', 'ClassProbabilityMax', 'ClassProbabilityMean', 'ClassProbabilityMin', 'ClassProbabilitySTD',
     'DatasetRatio', 'InverseDatasetRatio', 'LogDatasetRatio', 'LogInverseDatasetRatio', 'LogNumberOfFeatures',
     'LogNumberOfInstances', 'NumberOfCategoricalFeatures', 'NumberOfClasses', 'NumberOfFeatures',
     'NumberOfFeaturesWithMissingValues', 'NumberOfInstances', 'NumberOfInstancesWithMissingValues',
     'NumberOfMissingValues', 'NumberOfNumericFeatures', 'PercentageOfFeaturesWithMissingValues',
     'PercentageOfInstancesWithMissingValues', 'PercentageOfMissingValues', 'RatioNominalToNumerical',
     'RatioNumericalToNominal', 'SymbolsMax', 'SymbolsMean', 'SymbolsMin', 'SymbolsSTD', 'SymbolsSum']

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

mgen = SpaceGenerator()
mspace = mgen.generate_params()

my_list = list(mspace.name2node.keys())
my_list.sort()

my_list_constraints = ['global_search_time_constraint', 'global_evaluation_time_constraint', 'global_memory_constraint', 'global_cv', 'global_number_cv']


#print(my_list)

#test data

feature_names = copy.deepcopy(my_list)
feature_names.extend(my_list_constraints)
feature_names.extend(metafeature_names_new)

test_holdout_dataset_id = 1590#1218#4134#31#1139#31#1138#31
search_time_frozen = 10*60#20*60#240#240
memory_budget = 1.0

X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data(test_holdout_dataset_id, randomstate=42)
metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

model = pickle.load(open('/tmp/my_great_model.p', "rb"))
#model = pickle.load(open('/home/felix/phd2/my_meta_model/my_great_model.p', "rb"))

plot_most_important_features(model, feature_names, k=len(feature_names))



study_prune = optuna.create_study(direction='maximize')
study_prune.optimize(lambda trial: optimize_accuracy_under_constraints(trial, metafeature_values_hold, search_time_frozen, model, my_list, memory_budget), n_trials=500, n_jobs=4)

space = study_prune.best_trial.user_attrs['space']



for pre, _, node in RenderTree(space.parameter_tree):
    if node.status == True:
        print("%s%s" % (pre, node.name))

#search_time_frozen = 120


result, search = run_AutoML(study_prune.best_trial,
                                             X_train=X_train_hold,
                                             X_test=X_test_hold,
                                             y_train=y_train_hold,
                                             y_test=y_test_hold,
                                             categorical_indicator=categorical_indicator_hold,
                                             my_scorer=auc,
                                             search_time=search_time_frozen,
                                             memory_limit=memory_budget,
                                             cv=2,
                                             number_of_cvs=1)

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress
show_progress(search, X_test_hold, y_test_hold, auc)


print("test result: " + str(result))


