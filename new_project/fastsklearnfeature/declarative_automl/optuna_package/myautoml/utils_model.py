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
import matplotlib.pyplot as plt
import pickle
import operator
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations
import multiprocessing as mp

metafeature_names_new = ['ClassEntropy', 'ClassProbabilityMax', 'ClassProbabilityMean', 'ClassProbabilityMin', 'ClassProbabilitySTD',
     'DatasetRatio', 'InverseDatasetRatio', 'LogDatasetRatio', 'LogInverseDatasetRatio', 'LogNumberOfFeatures',
     'LogNumberOfInstances', 'NumberOfCategoricalFeatures', 'NumberOfClasses', 'NumberOfFeatures',
     'NumberOfFeaturesWithMissingValues', 'NumberOfInstances', 'NumberOfInstancesWithMissingValues',
     'NumberOfMissingValues', 'NumberOfNumericFeatures', 'PercentageOfFeaturesWithMissingValues',
     'PercentageOfInstancesWithMissingValues', 'PercentageOfMissingValues', 'RatioNominalToNumerical',
     'RatioNumericalToNominal', 'SymbolsMax', 'SymbolsMean', 'SymbolsMin', 'SymbolsSTD', 'SymbolsSum']

my_list_constraints = ['global_search_time_constraint',
                       'global_evaluation_time_constraint',
                       'global_memory_constraint',
                       'global_cv',
                       'global_number_cv',
                       'privacy',
                       'hold_out_fraction',
                       'sample_fraction',
                       'training_time_constraint',
                       'inference_time_constraint',
                       'pipeline_size_constraint']


mgen = SpaceGenerator()
mspace = mgen.generate_params()

my_list = list(mspace.name2node.keys())
my_list.sort()

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess


def get_feature_names():
    feature_names = copy.deepcopy(my_list)
    feature_names.extend(copy.deepcopy(my_list_constraints))
    feature_names.extend(copy.deepcopy(metafeature_names_new))

    feature_names_new = FeatureTransformations().get_new_feature_names(feature_names)
    return feature_names, feature_names_new

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

    #assert len(rf_random.feature_importances_) == len(names_features), 'mismatch'

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

def space2features(space, my_list_constraints_values, metafeature_values):
    tuple_param = np.zeros((1, len(my_list)))
    tuple_constraints = np.zeros((1, len(my_list_constraints)))
    t = 0
    for parameter_i in range(len(my_list)):
        tuple_param[t, parameter_i] = space.name2node[my_list[parameter_i]].status

    for constraint_i in range(len(my_list_constraints)):
        tuple_constraints[t, constraint_i] = my_list_constraints_values[constraint_i] #current_trial.params[my_list_constraints[constraint_i]]

    return np.hstack((tuple_param, tuple_constraints, metafeature_values))

def predict_range(model, X):
    y_pred = model.predict(X)

    #y_pred[y_pred > 1.0] = 1.0
    #y_pred[y_pred < 0.0] = 0.0
    return y_pred

def ifNull(value, constant_value=0):
    if type(value) == type(None):
        return constant_value
    else:
        return value






def optimize_accuracy_under_constraints2(trial, metafeature_values_hold, search_time, model_compare, model_success,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None
                                        ):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        if type(evaluation_time) == type(None):
            evaluation_time = search_time
            if trial.suggest_categorical('use_evaluation_time_constraint', [True, False]):
                evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        else:
            trial.set_user_attr('evaluation_time', evaluation_time)

        # how many cvs should be used
        cv = 1
        number_of_cvs = 1
        if type(hold_out_fraction) == type(None):
            hold_out_fraction = None
            if trial.suggest_categorical('use_hold_out', [True, False]):
                hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
            else:
                cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
                number_of_cvs = 1
                if trial.suggest_categorical('use_multiple_cvs', [True, False]):
                    number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        else:
            trial.set_user_attr('hold_out_fraction', hold_out_fraction)


        sample_fraction = 1.0
        if trial.suggest_categorical('use_sampling', [True, False]):
            sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)



        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(hold_out_fraction),
                                      sample_fraction,
                                      ifNull(training_time_limit, constant_value=search_time),
                                      ifNull(inference_time_limit, constant_value=60),
                                      ifNull(pipeline_size_limit, constant_value=350000000)]

        features = space2features(space, my_list_constraints_values, metafeature_values_hold)
        feature_names, _ = get_feature_names()
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)
        trial.set_user_attr('features', features)

        return predict_range(model_compare, features) + predict_range(model_success, features)
    except Exception as e:
        print(str(e) + 'except dataset _ accuracy: ' + '\n\n')
        return 0.0


def generate_parameters(trial, total_search_time, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 10, max(10, total_search_time), log=False)

    # how much time for each evaluation
    evaluation_time = search_time
    if trial.suggest_categorical('use_evaluation_time_constraint', [True, False]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', min(10, search_time), search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [True, False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [True, False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [True, False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [True, False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [True, False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True, False]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True, False]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id




def optimize_accuracy_under_constraints(trial, metafeature_values_hold, search_time, model,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        if type(evaluation_time) == type(None):
            evaluation_time = search_time
            if trial.suggest_categorical('use_evaluation_time_constraint', [True, False]):
                evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        else:
            trial.set_user_attr('evaluation_time', evaluation_time)

        # how many cvs should be used
        cv = 1
        number_of_cvs = 1
        if type(hold_out_fraction) == type(None):
            hold_out_fraction = None
            if trial.suggest_categorical('use_hold_out', [True, False]):
                hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
            else:
                cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
                number_of_cvs = 1
                if trial.suggest_categorical('use_multiple_cvs', [True, False]):
                    number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        else:
            trial.set_user_attr('hold_out_fraction', hold_out_fraction)


        sample_fraction = 1.0
        #if trial.suggest_categorical('use_sampling', [True, False]):
        #    sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)



        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(hold_out_fraction),
                                      sample_fraction]

        features = space2features(space, my_list_constraints_values, metafeature_values_hold)
        feature_names, _ = get_feature_names()
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)
        trial.set_user_attr('features', features)

        return predict_range(model, features)
    except Exception as e:
        print(str(e) + 'except dataset _ accuracy: ' + '\n\n')
        return 0.0

def run_AutoML(trial, X_train=None, X_test=None, y_train=None, y_test=None, categorical_indicator=None, my_scorer=None,
               search_time=None,
               memory_limit=None,
               privacy_limit=None,
               training_time_limit=None,
               inference_time_limit=None,
               pipeline_size_limit=None
               ):
    space = trial.user_attrs['space']

    print(trial.params)

    if 'evaluation_time' in trial.user_attrs:
        evaluation_time = trial.user_attrs['evaluation_time']
    else:
        evaluation_time = search_time
        if 'global_evaluation_time_constraint' in trial.params:
            evaluation_time = trial.params['global_evaluation_time_constraint']

    cv = 1
    number_of_cvs = 1
    if 'hold_out_fraction' in trial.user_attrs:
        hold_out_fraction = trial.user_attrs['hold_out_fraction']
    else:
        hold_out_fraction = None
        if 'global_cv' in trial.params:
            cv = trial.params['global_cv']
            if 'global_number_cv' in trial.params:
                number_of_cvs = trial.params['global_number_cv']
        if 'hold_out_fraction' in trial.params:
            hold_out_fraction = trial.params['hold_out_fraction']

    sample_fraction = 1.0
    if 'sample_fraction' in trial.params:
        sample_fraction = trial.params['sample_fraction']

    search = MyAutoML(cv=cv,
                      number_of_cvs=number_of_cvs,
                      n_jobs=1,
                      evaluation_budget=evaluation_time,
                      time_search_budget=search_time,
                      space=space,
                      main_memory_budget_gb=memory_limit,
                      differential_privacy_epsilon=privacy_limit,
                      hold_out_fraction=hold_out_fraction,
                      sample_fraction=sample_fraction,
                      training_time_limit=training_time_limit,
                      inference_time_limit=inference_time_limit,
                      pipeline_size_limit=pipeline_size_limit
                      )
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