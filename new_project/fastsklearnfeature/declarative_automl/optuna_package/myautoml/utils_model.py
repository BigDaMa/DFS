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
import operator

my_list_constraints = ['global_search_time_constraint', 'global_evaluation_time_constraint', 'global_memory_constraint', 'global_cv', 'global_number_cv']


metafeature_names_new = ['ClassEntropy', 'ClassProbabilityMax', 'ClassProbabilityMean', 'ClassProbabilityMin', 'ClassProbabilitySTD',
     'DatasetRatio', 'InverseDatasetRatio', 'LogDatasetRatio', 'LogInverseDatasetRatio', 'LogNumberOfFeatures',
     'LogNumberOfInstances', 'NumberOfCategoricalFeatures', 'NumberOfClasses', 'NumberOfFeatures',
     'NumberOfFeaturesWithMissingValues', 'NumberOfInstances', 'NumberOfInstancesWithMissingValues',
     'NumberOfMissingValues', 'NumberOfNumericFeatures', 'PercentageOfFeaturesWithMissingValues',
     'PercentageOfInstancesWithMissingValues', 'PercentageOfMissingValues', 'RatioNominalToNumerical',
     'RatioNumericalToNominal', 'SymbolsMax', 'SymbolsMean', 'SymbolsMin', 'SymbolsSTD', 'SymbolsSum']

def data2features(X_train, y_train, categorical_indicator):
    metafeatures = calculate_all_metafeatures_with_labels(X_train, y_train, categorical=categorical_indicator,
                                                          dataset_name='data')

    metafeature_values = np.zeros((1, len(metafeature_names_new)))
    for m_i in range(len(metafeature_names_new)):
        try:
            metafeature_values[0, m_i] = metafeatures[metafeature_names_new[m_i]].value
        except:
            pass
    return metafeature_values


def get_data(data_id, randomstate=42):
    dataset = openml.datasets.get_dataset(dataset_id=data_id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )



    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                                y,
                                                                                random_state=randomstate,
                                                                                stratify=y,
                                                                                train_size=0.6)

    return X_train, X_test, y_train, y_test, categorical_indicator, attribute_names


def plot_most_important_features(rf_random, names_features, title='importance', verbose=True, k=25):

    assert len(rf_random.feature_importances_) == len(names_features), 'mismatch'

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
        print(key + ': ' + str(value))
        t += 1
        if t == k:
            break

    ind = np.arange(len(score))
    plt.bar(ind, score, align='center', alpha=0.5)
    #plt.yticks(ind, labels)

    plt.xticks(ind, labels, rotation='vertical')

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.6)

    if verbose:
        plt.show()
    else:
        plt.savefig('/tmp/feature_importance.png')
        plt.clf()

def predict_range(model, X):
    y_pred = model.predict(X)

    y_pred[y_pred > 1.0] = 1.0
    y_pred[y_pred < 0.0] = 0.0
    return y_pred

def space2features(space, my_list_constraints_values, metafeature_values, my_list):
    tuple_param = np.zeros((1, len(my_list)))
    tuple_constraints = np.zeros((1, len(my_list_constraints)))
    t = 0
    for parameter_i in range(len(my_list)):
        tuple_param[t, parameter_i] = space.name2node[my_list[parameter_i]].status

    for constraint_i in range(len(my_list_constraints)):
        tuple_constraints[t, constraint_i] = my_list_constraints_values[constraint_i]

    return np.hstack((tuple_param, tuple_constraints, metafeature_values))

def optimize_accuracy_under_constraints(trial, metafeature_values_hold, search_time_frozen, model, my_list, memory_budget_gb):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time = search_time_frozen#trial.suggest_int('global_search_time_constraint', 10, search_time_frozen, log=False)
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        memory_limit = memory_budget_gb
        cv = trial.suggest_int('global_cv', 2, 20, log=False)
        number_of_cvs = trial.suggest_int('global_number_cv', 1, 10, log=False)

        my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]
        features = space2features(space, my_list_constraints_values, metafeature_values_hold, my_list)
        trial.set_user_attr('features', features)

        return predict_range(model, features)
    except Exception as e:
        print(str(e) + 'except dataset _ accuracy: ' + '\n\n')
        return 0.0

def run_AutoML(trial, X_train=None, X_test=None, y_train=None, y_test=None, categorical_indicator=None, my_scorer=None, search_time=None, memory_limit=None, cv=None, number_of_cvs=None):
    space = trial.user_attrs['space']

    print(trial.params)

    #make this a hyperparameter
    evaluation_time = trial.params['global_evaluation_time_constraint']

    if type(cv) == type(None):
        cv = trial.params['global_cv']
    if type(number_of_cvs) == type(None):
        number_of_cvs = trial.params['global_number_cv']

    if 'dataset_id' in trial.params:
        dataset_id = trial.params['dataset_id'] #get same random seed
    else:
        dataset_id = 31


    if type(X_train) == type(None):

        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id, randomstate=int(time.time()))

        if not isinstance(trial, FrozenTrial):
            my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]

            metafeature_values = data2features(X_train, y_train, categorical_indicator)
            features = space2features(space, my_list_constraints_values, metafeature_values)
            trial.set_user_attr('features', features)

    search = MyAutoML(cv=cv,
                      number_of_cvs=number_of_cvs,
                      n_jobs=1,
                      evaluation_budget=evaluation_time,
                      time_search_budget=search_time,
                      space=space,
                      main_memory_budget_gb=memory_limit)
    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

    best_pipeline = search.get_best_pipeline()

    test_score = 0.0
    if type(best_pipeline) != type(None):
        test_score = my_scorer(search.get_best_pipeline(), X_test, y_test)


    return test_score, search

def show_progress(search, X_test, y_test, scorer):
    times = []
    validation_scores = []
    best_scores = []
    current_best = 0
    real_scores = []
    current_real = 0.0

    for t in search.study.trials:
        try:
            current_time = t.user_attrs['time_since_start']
            current_val = t.value
            if current_val < 0:
                current_val = 0
            times.append(current_time)
            validation_scores.append(current_val)

            current_pipeline = None
            try:
                current_pipeline = t.user_attrs['pipeline']
            except:
                pass

            if type(current_pipeline) != type(None):
                current_real = scorer(current_pipeline, X_test, y_test)

            real_scores.append(copy.deepcopy(current_real))

            if current_val > current_best:
                current_best = current_val
            best_scores.append(copy.deepcopy(current_best))

        except:
            pass

    import matplotlib.pyplot as plt
    plt.plot(times, best_scores, color='red')
    plt.plot(times, real_scores, color='blue')
    plt.show()
    print(times)
    print(best_scores)
    print(real_scores)