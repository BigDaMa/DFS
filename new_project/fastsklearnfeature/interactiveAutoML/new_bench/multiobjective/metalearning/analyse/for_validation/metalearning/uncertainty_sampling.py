import copy
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import f_anova_wo


from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_mcfs
from sklearn.model_selection import train_test_split
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from skrebate import ReliefF
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.multiprocessing_global as mp_global
import diffprivlib.models as models
from sklearn.model_selection import GridSearchCV

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.fullfeatures import fullfeatures
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import TPE
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import simulated_annealing
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.evolution import evolution
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.exhaustive import exhaustive
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.recursive_feature_elimination import recursive_feature_elimination
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_fair_data1_validation
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.private_models.randomforest.PrivateRandomForrest import PrivateRandomForest


from sklearn.ensemble import RandomForestClassifier

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import os
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
import pathlib
import glob

def my_function(config_id):
	conf = mp_global.configurations[config_id]
	result = conf['main_strategy'](mp_global.X_train,
								   mp_global.X_validation,
								   mp_global.X_train_val,
								   mp_global.X_test,
								   mp_global.y_train,
								   mp_global.y_validation,
								   mp_global.y_train_val,
								   mp_global.y_test,
								   mp_global.names,
								   mp_global.sensitive_ids,
								   ranking_functions=conf['ranking_functions'],
								   clf=mp_global.clf,
								   min_accuracy=mp_global.min_accuracy,
								   min_fairness=mp_global.min_fairness,
								   min_robustness=mp_global.min_robustness,
								   max_number_features=mp_global.max_number_features,
								   max_search_time=mp_global.max_search_time,
								   log_file='/tmp/experiment_uncertainty/run' + str(0) + '/strategy' + str(conf['strategy_id']) + '.pickle',
								   accuracy_scorer=mp_global.accuracy_scorer
								   )
	result['strategy_id'] = conf['strategy_id']
	return result


def uncertainty_sampling(training_dataset_ids, all_current_models):
    #time_limit = 60 * 60 * 3

    training_dataset_ids = training_dataset_ids.tolist()

    if '1240' in training_dataset_ids:
        training_dataset_ids.remove('1240')
    if '42132' in training_dataset_ids:
        training_dataset_ids.remove('42132')

    time_limit = 60 * 10

    def maximize_uncertainty(hps):
        print(hps)

        X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, key, sensitive_attribute_id = get_fair_data1_validation(dataset_key=hps['data'])
        is_regression = False

        # run on tiny sample
        if X_train.shape[0] > 100:
            if is_regression:
                X_train_tiny, _, y_train_tiny, _ = train_test_split(X_train, y_train, train_size=100, random_state=42)
            else:
                X_train_tiny, _, y_train_tiny, _ = train_test_split(X_train, y_train, train_size=100, random_state=42,
                                                                    stratify=y_train)
        else:
            X_train_tiny = X_train
            y_train_tiny = y_train

        print(X_train.shape)

        if type(sensitive_ids) != type(None):
            fair_train_tiny = make_scorer(true_positive_rate_score, greater_is_better=True,
                                          sensitive_data=X_train_tiny[:, sensitive_ids[0]])

        mp_global.X_train = X_train
        mp_global.X_validation = X_validation
        mp_global.X_train_val = X_train_val
        mp_global.X_test = X_test
        mp_global.y_train = y_train
        mp_global.y_validation = y_validation
        mp_global.y_train_val = y_train_val
        mp_global.y_test = y_test
        mp_global.names = names
        mp_global.sensitive_ids = sensitive_ids

        mp_global.cv_splitter = StratifiedKFold(5, random_state=42)
        mp_global.accuracy_scorer = make_scorer(f1_score)

        mp_global.avoid_robustness = False


        cv_k = 1.0
        cv_privacy = hps['privacy']

        model = LogisticRegression(class_weight='balanced')
        if type(cv_privacy) != type(None):
            model = models.LogisticRegression(epsilon=cv_privacy, class_weight='balanced')

        if type(cv_privacy) == type(None):
            cv_privacy = X_train_tiny.shape[0]

        robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train_tiny, y=y_train_tiny, model=model,
                                    feature_selector=None, scorer=mp_global.accuracy_scorer)

        small_start_time = time.time()

        scoring = {'AUC': mp_global.accuracy_scorer}
        if not mp_global.avoid_robustness:
            scoring['Robustness'] = robust_scorer
        if type(sensitive_ids) != type(None):
            scoring['Fairness'] = fair_train_tiny

        cv = GridSearchCV(model, param_grid={}, scoring=scoring, refit=False, cv=mp_global.cv_splitter)
        cv.fit(X_train_tiny, pd.DataFrame(y_train_tiny))
        cv_acc = cv.cv_results_['mean_test_AUC'][0]

        cv_fair = 0.0
        if type(sensitive_ids) != type(None):
            cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]

        cv_robust = 0.0
        if not mp_global.avoid_robustness:
            cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

        cv_time = time.time() - small_start_time

        # construct feature vector
        feature_list = []
        # user-specified constraints
        feature_list.append(hps['accuracy'])
        feature_list.append(hps['fairness'])
        feature_list.append(hps['k'])
        feature_list.append(hps['k'] * X_train.shape[1])
        feature_list.append(hps['robustness'])
        feature_list.append(cv_privacy)
        feature_list.append(hps['search_time'])
        # differences to sample performance
        feature_list.append(cv_acc - hps['accuracy'])
        feature_list.append(cv_fair - hps['fairness'])
        feature_list.append(cv_k - hps['k'])
        feature_list.append((cv_k - hps['k']) * X_train.shape[1])
        feature_list.append(cv_robust - hps['robustness'])
        feature_list.append(cv_time)
        # privacy constraint is always satisfied => difference always zero => constant => unnecessary

        # metadata features
        feature_list.append(X_train.shape[0])  # number rows
        feature_list.append(X_train.shape[1])  # number columns

        #models
        feature_list.append(hps['model'] == 'Decision Tree')
        feature_list.append(hps['model'] == 'Gaussian Naive Bayes')
        feature_list.append(hps['model'] == 'Logistic Regression')

        features = np.array(feature_list).reshape(1, -1)

        # predict the best model and calculate uncertainty

        print(features)

        #now predict with models
        aggregated_certainty = 0
        print("uncertainty")
        for model_i in range(len(all_current_models)):
            certainty = np.abs(all_current_models[model_i].predict_proba(features)[0,0] - 0.5)
            aggregated_certainty += certainty


        print('Certainty: ' + str(aggregated_certainty))

        return {'loss': aggregated_certainty, 'status': STATUS_OK, 'features': features, 'search_time': hps['search_time'],
                'constraints': hps}

    space = {
        'data': hp.choice('data_choice', training_dataset_ids),
        'model': hp.choice('model_choice',
                           [
                               'Logistic Regression',
                               'Gaussian Naive Bayes',
                               'Decision Tree'  # , 'Random Forest'
                           ]),
        'k': hp.choice('k_choice',
                       [
                           (1.0),
                           (hp.uniform('k_specified', 0, 1))
                       ]),
        'accuracy': hp.uniform('accuracy_specified', 0.5, 1),
        'fairness': hp.choice('fairness_choice',
                              [
                                  (0.0),
                                  (hp.uniform('fairness_specified', 0.8, 1))
                              ]),
        'privacy': hp.choice('privacy_choice',
                             [
                                 (None),
                                 (hp.lognormal('privacy_specified', 0, 1))
                             ]),
        'robustness': hp.choice('robustness_choice',
                                [
                                    (0.0),
                                    (hp.uniform('robustness_specified', 0.8, 1))
                                ]),
        'search_time': hp.uniform('search_time_specified', 10, time_limit),  # in seconds
    }

    trials = Trials()
    fmin(maximize_uncertainty, space=space, algo=tpe.suggest, max_evals=100, trials=trials, show_progressbar=True)



    ### now run most uncertain trial

    number_of_runs = 1

    # break, once convergence tolerance is reached and generate new dataset
    last_trial = trials.best_trial
    most_uncertain_f = last_trial['misc']['vals']
    # print(most_uncertain_f)

    ##specifiy data
    run_counter = 0
    current_run_time_id = time.time()
    path = pathlib.Path('/tmp/experiment_uncertainty/run' + str(0))
    path.mkdir(parents=True, exist_ok=True)


    selected_dataset_id = training_dataset_ids[most_uncertain_f['data_choice'][0]]
    X_train, X_validation, X_train_val, X_test, y_train, y_validation, y_train_val, y_test, names, sensitive_ids, key, sensitive_attribute_id = get_fair_data1_validation(dataset_key=selected_dataset_id)
    is_regression = False

    mp_global.X_train = X_train
    mp_global.X_validation = X_validation
    mp_global.X_train_val = X_train_val
    mp_global.X_test = X_test
    mp_global.y_train = y_train
    mp_global.y_validation = y_validation
    mp_global.y_train_val = y_train_val
    mp_global.y_test = y_test
    mp_global.names = names
    mp_global.sensitive_ids = sensitive_ids

    if is_regression:
        mp_global.cv_splitter = KFold(5, random_state=42)
        mp_global.accuracy_scorer = make_scorer(r2_score)
    else:
        mp_global.cv_splitter = StratifiedKFold(5, random_state=42)
        mp_global.accuracy_scorer = make_scorer(f1_score)
    mp_global.avoid_robustness = False




    min_accuracy = most_uncertain_f['accuracy_specified'][0]
    min_fairness = 0.0
    if most_uncertain_f['fairness_choice'][0]:
        min_fairness = most_uncertain_f['fairness_specified'][0]
    min_robustness = 0.0
    if most_uncertain_f['robustness_choice'][0]:
        min_robustness = most_uncertain_f['robustness_specified'][0]
    max_number_features = 1.0
    if most_uncertain_f['k_choice'][0]:
        max_number_features = most_uncertain_f['k_specified'][0]

    max_search_time = most_uncertain_f['search_time_specified'][0]

    # Execute each search strategy with a given time limit (in parallel)
    # maybe run multiple times to smooth stochasticity

    model = None
    print(most_uncertain_f)
    if most_uncertain_f['model_choice'][0] == 0:
        model = LogisticRegression(class_weight='balanced')
        if most_uncertain_f['privacy_choice'][0]:
            model = models.LogisticRegression(epsilon=most_uncertain_f['privacy_specified'][0], class_weight='balanced')
    elif most_uncertain_f['model_choice'][0] == 1:
        model = GaussianNB()
        if most_uncertain_f['privacy_choice'][0]:
            model = models.GaussianNB(epsilon=most_uncertain_f['privacy_specified'][0])
    elif most_uncertain_f['model_choice'][0] == 2:
        model = DecisionTreeClassifier(class_weight='balanced')
        if most_uncertain_f['privacy_choice'][0]:
            model = PrivateRandomForest(n_estimators=1, epsilon=most_uncertain_f['privacy_specified'][0])
    elif most_uncertain_f['model_choice'][0] == 3:
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        if most_uncertain_f['privacy_choice'][0]:
            model = PrivateRandomForest(n_estimators=100, epsilon=most_uncertain_f['privacy_specified'][0])

    print(model)

    mp_global.clf = model
    # define rankings
    rankings = [variance,
                chi2_score_wo,
                fcbf,
                my_fisher_score,
                mutual_info_classif,
                my_mcfs]
    # rankings.append(partial(model_score, estimator=ExtraTreesClassifier(n_estimators=1000))) #accuracy ranking
    # rankings.append(partial(robustness_score, model=model, scorer=auc_scorer)) #robustness ranking
    # rankings.append(partial(fairness_score, estimator=ExtraTreesClassifier(n_estimators=1000), sensitive_ids=sensitive_ids)) #fairness ranking
    rankings.append(partial(model_score, estimator=ReliefF(n_neighbors=10)))  # relieff

    mp_global.min_accuracy = min_accuracy
    mp_global.min_fairness = min_fairness
    mp_global.min_robustness = min_robustness
    mp_global.max_number_features = max_number_features
    mp_global.max_search_time = max_search_time

    mp_global.configurations = []
    # add single rankings
    strategy_id = 1
    for r in range(len(rankings)):
        for run in range(number_of_runs):
            configuration = {}
            configuration['ranking_functions'] = copy.deepcopy([rankings[r]])
            configuration['run_id'] = copy.deepcopy(run)
            configuration['main_strategy'] = copy.deepcopy(weighted_ranking)
            configuration['strategy_id'] = copy.deepcopy(strategy_id)
            mp_global.configurations.append(configuration)
        strategy_id += 1

    main_strategies = [TPE,
                       simulated_annealing,
                       evolution,
                       exhaustive,
                       forward_selection,
                       backward_selection,
                       forward_floating_selection,
                       backward_floating_selection,
                       recursive_feature_elimination,
                       fullfeatures]

    # run main strategies

    for strategy in main_strategies:
        for run in range(number_of_runs):
            configuration = {}
            configuration['ranking_functions'] = []
            configuration['run_id'] = copy.deepcopy(run)
            configuration['main_strategy'] = copy.deepcopy(strategy)
            configuration['strategy_id'] = copy.deepcopy(strategy_id)
            mp_global.configurations.append(configuration)
        strategy_id += 1

    # 6#17
    with ProcessPool(max_workers=6) as pool:
        future = pool.map(my_function, range(len(mp_global.configurations)), timeout=max_search_time)

        iterator = future.result()
        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process

    #check which strategies were successful

    mappnames = {1: 'TPE(Variance)',
                 2: 'TPE($\chi^2$)',
                 3: 'TPE(FCBF)',
                 4: 'TPE(Fisher)',
                 5: 'TPE(MIM)',
                 6: 'TPE(MCFS)',
                 7: 'TPE(ReliefF)',
                 8: 'TPE(NR)',
                 9: 'SA(NR)',
                 10: 'NSGA-II(NR)',
                 11: 'ES(NR)',
                 12: 'SFS(NR)',
                 13: 'SBS(NR)',
                 14: 'SFFS(NR)',
                 15: 'SBFS(NR)',
                 16: 'RFE(LR)',
                 17: 'Complete Set'
                 }

    def load_pickle(fname):
        data = []
        with open(fname, "rb") as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        return data

    def is_successfull_validation_and_test(exp_results):
        return len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1][
            'success_test'] == True  # also on test satisfied

    def is_successfull_validation(exp_results):
        return len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[
            -1]  # constraints were satisfied on validation set

    run_strategies_success_test = {}
    run_strategies_times = {}
    run_strategies_success_validation = {}

    rfolder = '/tmp/experiment_uncertainty/run' + str(0) + '/'

    validation_satisfied_by_any_strategy = False

    min_time = np.inf
    best_strategy = 0
    for s in range(1, len(mappnames) + 1):
        exp_results = []
        try:
            exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
        except:
            pass
        if is_successfull_validation_and_test(exp_results):
            runtime = exp_results[-1]['final_time']
            if runtime < min_time:
                min_time = runtime
                best_strategy = s

            run_strategies_success_test[s] = True
            run_strategies_times[s] = runtime
        else:
            run_strategies_success_test[s] = False

        run_strategies_success_validation[s] = is_successfull_validation(exp_results)
        if run_strategies_success_validation[s]:
            validation_satisfied_by_any_strategy = True

    strategy_success = np.zeros((1, len(mappnames)))
    for c_i in range(len(mappnames)):
        strategy_success[0, c_i] = run_strategies_success_test[c_i + 1]

    return last_trial['result']['features'], strategy_success



