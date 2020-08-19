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


#test data

dataset_hold = openml.datasets.get_dataset(dataset_id=31)

X_hold, y_hold, categorical_indicator_hold, attribute_names_hold = dataset_hold.get_data(
    dataset_format='array',
    target=dataset_hold.default_target_attribute
)

X_train_hold, X_test_hold, y_train_hold, y_test_hold = sklearn.model_selection.train_test_split(X_hold, y_hold, random_state=42)



auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)



total_search_time = 120#10 * 60

my_openml_datasets = [  # 31, #German Credit
            #1464,  # Blood Transfusion
            #333,  # Monks Problem 1
            334,  # Monks Problem 2
            50,  # TicTacToe
            #1504,  # steel plates fault
            #3,  # kr-vs-kp
            #1494,  # qsar-biodeg
            #1510,  # wdbc
            #1489,  # phoneme
        ]


mgen = SpaceGenerator()
mspace = mgen.generate_params()

my_list = list(mspace.name2node.keys())
my_list.sort()

my_list_constraints = ['global_search_time_constraint', 'global_evaluation_time_constraint', 'global_memory_constraint', 'global_cv', 'global_number_cv']


metafeaturenn = ['ClassEntropy', 'ClassProbabilityMax', 'ClassProbabilityMean', 'ClassProbabilityMin', 'ClassProbabilitySTD', 'DatasetRatio', 'InverseDatasetRatio', 'LogDatasetRatio', 'LogInverseDatasetRatio', 'LogNumberOfFeatures', 'LogNumberOfInstances', 'NumberOfCategoricalFeatures', 'NumberOfClasses', 'NumberOfFeatures', 'NumberOfFeaturesWithMissingValues', 'NumberOfInstances', 'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues', 'NumberOfNumericFeatures', 'PercentageOfFeaturesWithMissingValues', 'PercentageOfInstancesWithMissingValues', 'PercentageOfMissingValues', 'RatioNominalToNumerical', 'RatioNumericalToNominal', 'SymbolsMax', 'SymbolsMean', 'SymbolsMin', 'SymbolsSTD', 'SymbolsSum']



def run_AutoML(trial, X_train=None, X_test=None, y_train=None, y_test=None, categorical_indicator=None):
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
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

        # how much memory is allowed
        memory_limit = trial.suggest_uniform('global_memory_constraint', 1.5, 4)

        # how many cvs should be used
        cv = trial.suggest_int('global_cv', 2, 20, log=False)

        number_of_cvs = trial.suggest_int('global_number_cv', 1, 10, log=False)

        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    else:
        space = trial.user_attrs['space']

        print(trial.params)

        #make this a hyperparameter
        search_time = trial.params['global_search_time_constraint']
        evaluation_time = trial.params['global_evaluation_time_constraint']
        memory_limit = trial.params['global_memory_constraint']
        cv = trial.params['global_cv']
        number_of_cvs = trial.params['global_number_cv']

        if 'dataset_id' in trial.params:
            dataset_id = trial.params['dataset_id'] #get same random seed
        else:
            dataset_id = 31


    for pre, _, node in RenderTree(space.parameter_tree):
        print("%s%s: %s" % (pre, node.name, node.status))

    # which dataset to use
    #todo: add more datasets


    if type(X_train) == type(None):

        dataset = openml.datasets.get_dataset(dataset_id=dataset_id)

        print(dataset.name)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='array',
            target=dataset.default_target_attribute
        )

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=int(time.time()))

        if not isinstance(trial, FrozenTrial):
            my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]
            features = space2features(space, my_list_constraints_values, X_train, y_train, categorical_indicator)
            trial.set_user_attr('features', features)

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


def data2features(X_train, y_train, categorical_indicator):
    metafeatures = calculate_all_metafeatures_with_labels(X_train, y_train, categorical=categorical_indicator,
                                                          dataset_name='data')

    metafeature_names = list(metafeatures.keys())
    metafeature_names.sort()

    metafeature_names_new = []
    for n_i in range(len(metafeature_names)):
        if metafeatures[metafeature_names[n_i]].type_ == "METAFEATURE":
            metafeature_names_new.append(metafeature_names[n_i])
    metafeature_names_new.sort()
    #print(metafeature_names_new)

    metafeature_values = np.zeros((1, len(metafeature_names_new)))
    for m_i in range(len(metafeature_names_new)):
        metafeature_values[0, m_i] = metafeatures[metafeature_names_new[m_i]].value
    return metafeature_values


def space2features(space, my_list_constraints_values, X_train, y_train, categorical_indicator):
    tuple_param = np.zeros((1, len(my_list)))
    tuple_constraints = np.zeros((1, len(my_list_constraints)))
    t = 0
    for parameter_i in range(len(my_list)):
        tuple_param[t, parameter_i] = space.name2node[my_list[parameter_i]].status


    for constraint_i in range(len(my_list_constraints)):
        tuple_constraints[t, constraint_i] = my_list_constraints_values[constraint_i] #current_trial.params[my_list_constraints[constraint_i]]

    metafeature_values = data2features(X_train, y_train, categorical_indicator)

    return np.hstack((tuple_param, tuple_constraints, metafeature_values))

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
    evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
    memory_limit = trial.suggest_uniform('global_memory_constraint', 1.5, 4)
    cv = trial.suggest_int('global_cv', 2, 20, log=False)
    number_of_cvs = trial.suggest_int('global_number_cv', 1, 10, log=False)

    dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    dataset = openml.datasets.get_dataset(dataset_id=dataset_id)

    print(dataset.name)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    random_seed = int(time.time())

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=random_seed)

    #add metafeatures of data


    my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]
    features = space2features(space, my_list_constraints_values, X_train, y_train, categorical_indicator)
    trial.set_user_attr('features', features)

    predictions = []
    for tree in range(model.n_estimators):
        predictions.append(predict_range(model.estimators_[tree], features))

    stddev_pred = np.std(np.matrix(predictions).transpose(), axis=1)

    return stddev_pred[0]

search_time_frozen = 60

def optimize_accuracy_under_constraints(trial, X_train, y_train, categorical_indicator):
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
    features = space2features(space, my_list_constraints_values, X_train, y_train, categorical_indicator)
    trial.set_user_attr('features', features)

    return predict_range(model, features)

'''
def trial2features(trial, X_train, y_train, categorical_indicator):
    X_row_params = np.zeros((1, len(my_list)))
    X_row_constraints = np.zeros((1, len(my_list_constraints)))

    current_trial = trial
    t = 0
    for parameter_i in range(len(my_list)):
        X_row_params[t, parameter_i] = current_trial.user_attrs['space'].name2node[my_list[parameter_i]].status

    for constraint_i in range(len(my_list_constraints)):
        X_row_constraints[t, constraint_i] = current_trial.params[my_list_constraints[constraint_i]]

    metafeature_values = data2features(X_train, y_train, categorical_indicator)


    X_row_all = np.hstack((X_row_params, X_row_constraints, metafeature_values))

    return X_row_all
'''




#random sampling 10 iterations
study = optuna.create_study(direction='maximize', sampler=RandomSampler(seed=42))


#first random sampling
study.optimize(run_AutoML, n_trials=4, n_jobs=2)

print('done')

#generate training data
all_trials = study.get_trials()
X_meta = []
y_meta = []

#todo: create metafeatures for dataset
#todo: add log scaled search time

for t in range(len(study.get_trials())):
    current_trial = all_trials[t]
    if current_trial.value >= 0 and current_trial.value <= 1.0:
        y_meta.append(current_trial.value)
    else:
        y_meta.append(0.0)
    X_meta.append(current_trial.user_attrs['features'])

X_meta = np.vstack(X_meta)
print(X_meta.shape)

feature_names = copy.deepcopy(my_list)
feature_names.extend(my_list_constraints)
feature_names.extend(metafeaturenn)


pruned_accuray_results = []

while True:

    model = RandomForestRegressor()
    model.fit(X_meta, y_meta)

    #random sampling 10 iterations
    study_uncertainty = optuna.create_study(direction='maximize')
    study_uncertainty.optimize(optimize_uncertainty, n_trials=100, n_jobs=4)

    X_meta = np.vstack((X_meta, study_uncertainty.best_trial.user_attrs['features']))
    y_meta.append(run_AutoML(study_uncertainty.best_trial))

    study_prune = optuna.create_study(direction='maximize')
    study_prune.optimize(lambda trial: optimize_accuracy_under_constraints(trial, X_train_hold, y_train_hold, categorical_indicator_hold), n_trials=500, n_jobs=4)

    pruned_accuray_results.append(run_AutoML(study_prune.best_trial,
                                             X_train=X_train_hold,
                                             X_test=X_test_hold,
                                             y_train=y_train_hold,
                                             y_test=y_test_hold,
                                             categorical_indicator=categorical_indicator_hold))

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
