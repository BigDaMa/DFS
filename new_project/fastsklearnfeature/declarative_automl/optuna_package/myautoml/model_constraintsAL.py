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
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features

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



#test data

test_holdout_dataset_id = 31
search_time_frozen = 120

X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data(test_holdout_dataset_id, randomstate=42)


metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)



total_search_time = 60*60#10 * 60

my_openml_datasets = [3, 4, 13, 15, 24, 25, 29, 31, 37, 38, 40, 43, 44, 49, 50, 51, 52, 53, 55, 56, 59, 151, 152, 153, 161, 162, 164, 172, 179, 310, 311, 312, 316, 333, 334, 335, 336, 337, 346, 444, 446, 448, 450, 451, 459, 461, 463, 464, 465, 466, 467, 470, 472, 476, 479, 481, 682, 683, 747, 803, 981, 993, 1037, 1038, 1039, 1040, 1042, 1045, 1046, 1048, 1049, 1050, 1053, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073, 1075, 1085, 1101, 1104, 1107, 1111, 1112, 1114, 1116, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1169, 1216, 1235, 1236, 1237, 1238, 1240, 1412, 1441, 1442, 1443, 1444, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1455, 1458, 1460, 1461, 1462, 1463, 1464, 1467, 1471, 1473, 1479, 1480, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1494, 1495, 1496, 1498, 1502, 1504, 1506, 1507, 1510, 1511, 1547, 1561, 1562, 1563, 1564, 1597, 4134, 4135, 4154, 4329, 4534, 23499, 40536, 40645, 40646, 40647, 40648, 40649, 40650, 40660, 40665, 40666, 40669, 40680, 40681, 40690, 40693, 40701, 40705, 40706, 40710, 40713, 40714, 40900, 40910, 40922, 40999, 41005, 41007, 41138, 41142, 41144, 41145, 41146, 41147, 41150, 41156, 41158, 41159, 41160, 41161, 41162, 41228, 41430, 41521, 41538, 41976, 42172, 42477]
my_openml_datasets.remove(test_holdout_dataset_id)


mgen = SpaceGenerator()
mspace = mgen.generate_params()

my_list = list(mspace.name2node.keys())
my_list.sort()

my_list_constraints = ['global_search_time_constraint', 'global_evaluation_time_constraint', 'global_memory_constraint', 'global_cv', 'global_number_cv']


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
        cv = trial.suggest_int('global_cv', 2, 20, log=False) #todo: calculate minimum number of splits based on y

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
    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

    best_pipeline = search.get_best_pipeline()

    test_score = 0.0
    if type(best_pipeline) != type(None):
        test_score = auc(search.get_best_pipeline(), X_test, y_test)


    return test_score



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

    y_pred[y_pred > 1.0] = 1.0
    y_pred[y_pred < 0.0] = 0.0
    return y_pred

def optimize_uncertainty(trial):
    try:
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

        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id,
                                                                                            randomstate=int(time.time()))

        #add metafeatures of data


        my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]

        metafeature_values = data2features(X_train, y_train, categorical_indicator)
        features = space2features(space, my_list_constraints_values, metafeature_values)

        trial.set_user_attr('features', features)

        predictions = []
        for tree in range(model.n_estimators):
            predictions.append(predict_range(model.estimators_[tree], features))

        stddev_pred = np.std(np.matrix(predictions).transpose(), axis=1)

        return stddev_pred[0]
    except Exception as e:
        print(str(e) + 'except dataset _ uncertainty: ' + str(dataset_id) + '\n\n')
        return 0.0

def optimize_accuracy_under_constraints(trial, metafeature_values_hold): #todo: transfer use features directly
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time = trial.suggest_int('global_search_time_constraint', 10, search_time_frozen, log=False)
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        memory_limit = trial.suggest_uniform('global_memory_constraint', 0.001, 4)
        cv = trial.suggest_int('global_cv', 2, 20, log=False)
        number_of_cvs = trial.suggest_int('global_number_cv', 1, 10, log=False)

        my_list_constraints_values = [search_time, evaluation_time, memory_limit, cv, number_of_cvs]
        features = space2features(space, my_list_constraints_values, metafeature_values_hold)
        trial.set_user_attr('features', features)

        return predict_range(model, features)
    except Exception as e:
        print(str(e) + 'except dataset _ accuracy: ' + '\n\n')
        return 0.0



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
feature_names.extend(metafeature_names_new)


pruned_accuray_results = []

verbose = False

while True:

    model = RandomForestRegressor()
    model.fit(X_meta, y_meta)

    with open('/tmp/my_great_model.p', "wb") as pickle_model_file:
        pickle.dump(model, pickle_model_file)

    #random sampling 10 iterations
    study_uncertainty = optuna.create_study(direction='maximize')
    study_uncertainty.optimize(optimize_uncertainty, n_trials=100, n_jobs=1) #todo: maybe wrap it into a process so it wont be killed by out of memory

    X_meta = np.vstack((X_meta, study_uncertainty.best_trial.user_attrs['features'])) #todo: add features for all better validation scores
    y_meta.append(run_AutoML(study_uncertainty.best_trial))

    study_prune = optuna.create_study(direction='maximize')
    study_prune.optimize(lambda trial: optimize_accuracy_under_constraints(trial, metafeature_values_hold), n_trials=500, n_jobs=4)

    pruned_accuray_results.append(run_AutoML(study_prune.best_trial,
                                             X_train=X_train_hold,
                                             X_test=X_test_hold,
                                             y_train=y_train_hold,
                                             y_test=y_test_hold,
                                             categorical_indicator=categorical_indicator_hold))

    plt.plot(range(len(pruned_accuray_results)), pruned_accuray_results)
    if verbose:
        plt.show()
    else:
        plt.savefig('/tmp/example_performance.png')
        plt.clf()
        print("Results on check")
        print(pruned_accuray_results)



    print('Shape: ' + str(X_meta.shape))

    plot_most_important_features(model, feature_names, verbose=verbose)
