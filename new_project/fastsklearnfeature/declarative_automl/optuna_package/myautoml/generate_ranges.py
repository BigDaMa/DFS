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
from optuna.samplers import RandomSampler
import matplotlib.pyplot as plt
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
import multiprocessing as mp
import heapq
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.mp_global_vars as mp_glob
from sklearn.metrics import f1_score
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import space2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import predict_range
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import MyPool
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import ifNull


#test data

test_holdout_dataset_id = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]
#X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data(test_holdout_dataset_id, randomstate=42)
#metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

#auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
my_scorer = make_scorer(f1_score)


total_search_time = 60*60#60*60#60*60#10 * 60

my_openml_datasets = [3, 4, 13, 15, 24, 25, 29, 31, 37, 38, 40, 43, 44, 49, 50, 51, 52, 53, 55, 56, 59, 151, 152, 153, 161, 162, 164, 172, 179, 310, 311, 312, 316, 333, 334, 335, 336, 337, 346, 444, 446, 448, 450, 451, 459, 461, 463, 464, 465, 466, 467, 470, 472, 476, 479, 481, 682, 683, 747, 803, 981, 993, 1037, 1038, 1039, 1040, 1042, 1045, 1046, 1048, 1049, 1050, 1053, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073, 1075, 1085, 1101, 1104, 1107, 1111, 1112, 1114, 1116, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1169, 1216, 1235, 1236, 1237, 1238, 1240, 1412, 1441, 1442, 1443, 1444, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1455, 1458, 1460, 1461, 1462, 1463, 1464, 1467, 1471, 1473, 1479, 1480, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1494, 1495, 1496, 1498, 1502, 1504, 1506, 1507, 1510, 1511, 1547, 1561, 1562, 1563, 1564, 1597, 4134, 4135, 4154, 4329, 4534, 23499, 40536, 40645, 40646, 40647, 40648, 40649, 40650, 40660, 40665, 40666, 40669, 40680, 40681, 40690, 40693, 40701, 40705, 40706, 40710, 40713, 40714, 40900, 40910, 40922, 40999, 41005, 41007, 41138, 41142, 41144, 41145, 41146, 41147, 41150, 41156, 41158, 41159, 41160, 41161, 41162, 41228, 41430, 41521, 41538, 41976, 42172, 42477]
for t_v in test_holdout_dataset_id:
    my_openml_datasets.remove(t_v)


feature_names, feature_names_new = get_feature_names()

def generate_parameters(trial):
    search_time = total_search_time
    evaluation_time = search_time
    memory_limit = 4
    privacy_limit = None

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

    dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id


def run_AutoML(trial, X_train=None, X_test=None, y_train=None, y_test=None, categorical_indicator=None):
    space = None
    search_time = None
    if not 'space' in trial.user_attrs:
        # which hyperparameters to use
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time, evaluation_time, memory_limit, privacy_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id = generate_parameters(trial)

    else:
        space = trial.user_attrs['space']

        print(trial.params)

        #make this a hyperparameter
        search_time = total_search_time
        evaluation_time = search_time
        memory_limit = 4
        privacy_limit = None

        cv = 1
        number_of_cvs = 1
        hold_out_fraction = None
        if 'global_cv' in trial.params:
            cv = trial.params['global_cv']
            if 'global_number_cv' in trial.params:
                number_of_cvs = trial.params['global_number_cv']
        else:
            hold_out_fraction = trial.params['hold_out_fraction']

        sample_fraction = 1.0
        if 'sample_fraction' in trial.params:
            sample_fraction = trial.params['sample_fraction']

        if 'dataset_id' in trial.params:
            dataset_id = trial.params['dataset_id'] #get same random seed
        else:
            dataset_id = 31

    for pre, _, node in RenderTree(space.parameter_tree):
        if node.status == True:
            print("%s%s" % (pre, node.name))

    if type(X_train) == type(None):

        my_random_seed = int(time.time())
        if 'data_random_seed' in trial.user_attrs:
            my_random_seed = trial.user_attrs['data_random_seed']

        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id, randomstate=my_random_seed)

        if not isinstance(trial, FrozenTrial):
            my_list_constraints_values = [search_time,
                                          evaluation_time,
                                          memory_limit, cv,
                                          number_of_cvs,
                                          ifNull(privacy_limit, constant_value=1000),
                                          ifNull(hold_out_fraction),
                                          sample_fraction]

            metafeature_values = data2features(X_train, y_train, categorical_indicator)
            features = space2features(space, my_list_constraints_values, metafeature_values)
            features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)
            trial.set_user_attr('features', features)

    search = MyAutoML(cv=cv,
                      number_of_cvs=number_of_cvs,
                      n_jobs=1,
                      evaluation_budget=evaluation_time,
                      time_search_budget=search_time,
                      space=space,
                      main_memory_budget_gb=memory_limit,
                      differential_privacy_epsilon=privacy_limit,
                      hold_out_fraction=hold_out_fraction,
                      sample_fraction=sample_fraction)
    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

    best_pipeline = search.get_best_pipeline()

    test_score = 0.0
    if type(best_pipeline) != type(None):
        test_score = my_scorer(search.get_best_pipeline(), X_test, y_test)


    return test_score, search


def run_AutoML_global(trial_id):
    try:
        _, current_search = run_AutoML(mp_glob.my_trials[trial_id])
    except:
        current_search = None

    training_times = []
    inference_times = []
    pipeline_sizes = []

    if type(current_search) != type(None):
        for my_trial in current_search.study.trials:
            try:
                if 'training_time' in my_trial.user_attrs:
                    training_times.append(my_trial.user_attrs['training_time'])
                if 'inference_time' in my_trial.user_attrs:
                    inference_times.append(my_trial.user_attrs['inference_time'])
                if 'pipeline_size' in my_trial.user_attrs:
                    pipeline_sizes.append(my_trial.user_attrs['pipeline_size'])
            except:
                pass

    return {'training_times': training_times, 'inference_times': inference_times, 'pipeline_sizes': pipeline_sizes}



def optimize_uncertainty(trial):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time, evaluation_time, memory_limit, privacy_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id = generate_parameters(
            trial)

        my_random_seed = int(time.time())
        trial.set_user_attr('data_random_seed', my_random_seed)

        return 0
    except Exception as e:
        print(str(e) + 'except dataset _ uncertainty: ' + str(dataset_id) + '\n\n')
        return 0.0




pruned_accuray_results = []

verbose = False

loss_over_time = []

training_times_all = []
inference_times_all = []
pipeline_sizes_all = []

while True:

    #random sampling 10 iterations
    study_uncertainty = optuna.create_study(direction='maximize', sampler=RandomSampler())
    study_uncertainty.optimize(optimize_uncertainty, n_trials=100, n_jobs=1) #todo: maybe wrap it into a process so it wont be killed by out of memory

    #get most uncertain for k datasets and run k runs in parallel
    topk = 20#20
    data2most_uncertain = {}
    for u_trial in study_uncertainty.trials:
        u_dataset = u_trial.params['dataset_id']
        u_value = u_trial.value
        data2most_uncertain[u_dataset] = (u_trial, u_value)

    k_keys_sorted_by_values = heapq.nlargest(topk, data2most_uncertain, key=lambda s: data2most_uncertain[s][1])

    mp_glob.my_trials = []
    for keyy in k_keys_sorted_by_values:
        mp_glob.my_trials.append(data2most_uncertain[keyy][0])

    with MyPool(processes=topk) as pool:
        results = pool.map(run_AutoML_global, range(topk))

        for r in results:
            training_times_all.extend(r['training_times'])
            inference_times_all.extend(r['inference_times'])
            pipeline_sizes_all.extend(r['pipeline_sizes'])

    new_dict = {}
    new_dict['training_times'] = training_times_all
    new_dict['inference_times'] = inference_times_all
    new_dict['pipeline_sizes'] = pipeline_sizes_all

    '''
    plt.hist(new_dict['training_times'], bins=100)
    print(len(new_dict['training_times']))
    plt.show()

    plt.hist(new_dict['inference_times'], bins=100)
    print(len(new_dict['inference_times']))
    plt.show()

    plt.hist(new_dict['pipeline_sizes'], bins=100)
    print(len(new_dict['pipeline_sizes']))
    print(new_dict['pipeline_sizes'])
    plt.show()
    '''

    pickle.dump(new_dict, open("/tmp/data.p", "wb"))


