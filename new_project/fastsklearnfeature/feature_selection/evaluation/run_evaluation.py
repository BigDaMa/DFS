from typing import List, Dict, Set
import numpy as np
from fastsklearnfeature.configuration.Config import Config
from fastsklearnfeature.candidates.RawFeature import RawFeature
import time
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import itertools
import warnings
import tqdm
import multiprocessing as mp
import fastsklearnfeature.feature_selection.evaluation.my_globale_module as my_globale_module
from fastsklearnfeature.transformations.MinusTransformation import MinusTransformation
from fastsklearnfeature.transformations.HigherOrderCommutativeTransformation import HigherOrderCommutativeTransformation
from fastsklearnfeature.transformations.binary.NonCommutativeBinaryTransformation import NonCommutativeBinaryTransformation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
import copy
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
def run_10_folds(classifier, train_transformed, target_train_folds, test_transformed, target_test_folds,score, C):
    parameter_set = {'penalty': 'l2', 'solver': 'lbfgs','class_weight': 'balanced','max_iter': 10000,'multi_class':'auto'}
    parameter_set['C'] = C
    score_list = []

    for fold in range(len(train_transformed)):
        clf = classifier(**parameter_set)
        clf.fit(train_transformed[fold], target_train_folds[fold])
        score_list.append(score(clf, test_transformed[fold], target_test_folds[fold]))

    return {'loss':1 - np.mean(score_list), 'status': STATUS_OK}

def hyperopt_search(train_transformed, test_transformed, training_all, one_test_set_transformed,
                grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target, test_target):

    trials = Trials()
    best = fmin(fn=lambda C: run_10_folds(classifier, train_transformed, target_train_folds, test_transformed, target_test_folds,score, C), space=hp.loguniform('C', np.log(1e-5), np.log(1e5)), algo=tpe.suggest,
                max_evals=100, trials=trials)
    print(best)
    #print(trials.results)


    best_param = {'penalty': 'l2', 'solver': 'lbfgs','class_weight': 'balanced','max_iter': 10000,'multi_class':'auto'}
    best_param['C'] = best['C']
    print(best_param)
    best_mean_cross_val_score = (-1 * trials.best_trial['result']['loss']) + 1.0
    best_score_list = []
    best_test_fold_predictions = []
    mean_scores = []

    additional_metric = {}

    clf = None

    test_score = -1
    if type(training_all) != type(None):
        # refit to entire training and test on test set
        clf = classifier(**best_param)
        clf.fit(training_all, train_y_all_target)
        #y_pred = clf.predict(one_test_set_transformed) #toberemoved
        #test_score = score._sign * score._score_func(test_target, y_pred, **score._kwargs)
        test_score = score(clf, one_test_set_transformed, test_target)
        #print('test: ' + str(test_score))

        #np.save('/tmp/true_predictions', self.test_target)


    return best_mean_cross_val_score, test_score, best_param, best_test_fold_predictions, best_score_list, clf, mean_scores, additional_metric

def calculate_likelyhood(model, X, y):
    one_class_index = -1
    for class_index in range(len(model.classes_)):
        if model.classes_[class_index] == 1:
            one_class_index = class_index
            break

    y_hat = model.predict_proba(X)[:, one_class_index]
    # y_hat = model.predict(X)
    resid = y - y_hat
    sse = sum(resid ** 2)
    return sse

#https://www.reddit.com/r/statistics/comments/5h2rbw/does_python_have_a_package_for_aicbic/
def calculate_AIC_for_classification(model, X, y, k):
    L = calculate_likelyhood(model, X, y)
    AIC = 2 * k - 2 * np.log(L)
    return AIC

def calculate_BIC_for_classification(model, X, y, k):
    L = calculate_likelyhood(model, X, y)
    n = X.shape[0]
    BIC = np.log(n) * k - 2 * np.log(L)
    return BIC

def calculate_AICc_for_classification(model, X, y, k):
    n = X.shape[0]
    AIC = calculate_AIC_for_classification(model, X, y, k)

    AICc = AIC + ((2*k**2 + 2*k)/(n-k-1))
    return AICc

def calculate_consistency(X,y):
    key2labels = {}

    for row_id in range(X.shape[0]):
        key = tuple(X[row_id])
        if not key in key2labels:
            key2labels[key] = set()
        key2labels[key].add(y[row_id])

    inconsistent_values = set()
    for k, v in key2labels.items():
        if len(v) > 1:
            inconsistent_values.add(k)

    consistency_counter = 0
    for row_id in range(X.shape[0]):
        key = tuple(X[row_id])
        if not key in inconsistent_values:
            consistency_counter += 1

    return float(consistency_counter) / X.shape[0]



def grid_search(candidate, train_transformed, test_transformed, training_all, one_test_set_transformed,
                grid_search_parameters, score, classifier, target_train_folds, target_test_folds, train_y_all_target, test_target):

    hyperparam_to_score_list = {}

    hyperparam_to_additional_score_list = {}

    my_keys = list(grid_search_parameters.keys())

    test_fold_predictions = {}

    for parameter_combination in itertools.product(*[grid_search_parameters[k] for k in my_keys]):
        parameter_set = hashabledict(zip(my_keys, parameter_combination))
        hyperparam_to_score_list[parameter_set] = []
        test_fold_predictions[parameter_set] = []

        hyperparam_to_additional_score_list[parameter_set] = {}
        hyperparam_to_additional_score_list[parameter_set]['accuracy'] = []
        hyperparam_to_additional_score_list[parameter_set]['f1'] = []

        hyperparam_to_additional_score_list[parameter_set]['AIC_feature_number'] = []
        hyperparam_to_additional_score_list[parameter_set]['AICc_feature_number'] = []
        hyperparam_to_additional_score_list[parameter_set]['BIC_feature_number'] = []

        hyperparam_to_additional_score_list[parameter_set]['AIC_complexity'] = []
        hyperparam_to_additional_score_list[parameter_set]['AICc_complexity'] = []
        hyperparam_to_additional_score_list[parameter_set]['BIC_complexity'] = []

        hyperparam_to_additional_score_list[parameter_set]['consistency'] = []


        for fold in range(len(train_transformed)):
            clf = classifier(**parameter_set)
            clf.fit(train_transformed[fold], target_train_folds[fold])
            y_pred = clf.predict(test_transformed[fold]) #toberemoved
            test_fold_predictions[parameter_set].append(y_pred == target_test_folds[fold]) #toberemoved
            #hyperparam_to_score_list[parameter_set].append(score._sign * score._score_func(target_test_folds[fold], y_pred, **score._kwargs))
            hyperparam_to_score_list[parameter_set].append(score(clf, test_transformed[fold], target_test_folds[fold]))

            ##add new scores
            hyperparam_to_additional_score_list[parameter_set]['accuracy'].append(make_scorer(accuracy_score)(clf, test_transformed[fold], target_test_folds[fold]))
            hyperparam_to_additional_score_list[parameter_set]['f1'].append(make_scorer(f1_score)(clf, test_transformed[fold], target_test_folds[fold]))

            k = test_transformed[fold].shape[1]
            hyperparam_to_additional_score_list[parameter_set]['AIC_feature_number'].append(calculate_AIC_for_classification(clf, test_transformed[fold], target_test_folds[fold], k))
            hyperparam_to_additional_score_list[parameter_set]['AICc_feature_number'].append(calculate_AICc_for_classification(clf, test_transformed[fold], target_test_folds[fold], k))
            hyperparam_to_additional_score_list[parameter_set]['BIC_feature_number'].append(calculate_BIC_for_classification(clf, test_transformed[fold], target_test_folds[fold], k))

            k = candidate.get_complexity()
            hyperparam_to_additional_score_list[parameter_set]['AIC_complexity'].append(calculate_AIC_for_classification(clf, test_transformed[fold], target_test_folds[fold], k))
            hyperparam_to_additional_score_list[parameter_set]['AICc_complexity'].append(calculate_AICc_for_classification(clf, test_transformed[fold], target_test_folds[fold], k))
            hyperparam_to_additional_score_list[parameter_set]['BIC_complexity'].append(calculate_BIC_for_classification(clf, test_transformed[fold], target_test_folds[fold], k))

            hyperparam_to_additional_score_list[parameter_set]['consistency'].append(calculate_consistency(test_transformed[fold], target_test_folds[fold]))


    best_param = None
    best_mean_cross_val_score = -float("inf")
    best_score_list = []
    best_test_fold_predictions = []
    mean_scores = []

    additional_metric = {}

    for parameter_config, score_list in hyperparam_to_score_list.items():
        #mean_score = np.min(score_list)
        mean_score = np.mean(score_list)
        mean_scores.append(mean_score)
        if mean_score > best_mean_cross_val_score:
            best_param = parameter_config
            best_mean_cross_val_score = mean_score
            best_score_list = copy.deepcopy(score_list)
            best_test_fold_predictions = test_fold_predictions[parameter_config]

            for k, metric_list in hyperparam_to_additional_score_list[parameter_config].items():
                additional_metric[k] = np.mean(metric_list)

    test_score = -1
    if type(training_all) != type(None):
        # refit to entire training and test on test set
        clf = classifier(**best_param)
        clf.fit(training_all, train_y_all_target)
        #y_pred = clf.predict(one_test_set_transformed) #toberemoved
        #test_score = score._sign * score._score_func(test_target, y_pred, **score._kwargs)
        test_score = score(clf, one_test_set_transformed, test_target)
        #print('test: ' + str(test_score))

        #np.save('/tmp/true_predictions', self.test_target)


    return best_mean_cross_val_score, test_score, best_param, best_test_fold_predictions, best_score_list, clf, mean_scores, additional_metric



def evaluate(candidate_id: int):
    #process = psutil.Process(os.getpid())
    #print(str(process.memory_info().rss) + " " + str(sys.getsizeof(my_globale_module.candidate_list_global)))

    candidate: CandidateFeature = my_globale_module.candidate_list_global[candidate_id]

    if type(my_globale_module.max_timestamp_global) != type(None) and time.time() >= my_globale_module.max_timestamp_global:
        raise RuntimeError('Out of time!')

    train_transformed = [None] * len(my_globale_module.preprocessed_folds_global)
    test_transformed = [None] * len(my_globale_module.preprocessed_folds_global)

    #test
    training_all = None
    one_test_set_transformed = None

    if isinstance(candidate, RawFeature):

        if 'training_all' in candidate.runtime_properties:
            training_all = candidate.runtime_properties['training_all']
            one_test_set_transformed = candidate.runtime_properties['one_test_set_transformed']

        train_transformed = candidate.runtime_properties['train_transformed']
        test_transformed = candidate.runtime_properties['test_transformed']

    else:

        #print(self.name_to_train_transformed.keys())

        #merge columns from parents
        test_unique_values = 0
        for fold in range(len(my_globale_module.preprocessed_folds_global)):
            train_transformed_input = np.hstack([p.runtime_properties['train_transformed'][fold] for p in candidate.parents])
            test_transformed_input = np.hstack([p.runtime_properties['test_transformed'][fold] for p in candidate.parents])

            candidate.transformation.fit(train_transformed_input, my_globale_module.target_train_folds_global[fold])
            train_transformed[fold] = candidate.transformation.transform(train_transformed_input)
            test_transformed[fold] = candidate.transformation.transform(test_transformed_input)
            test_unique_values += len(np.unique(test_transformed[fold]))

        if not isinstance(candidate.transformation, IdentityTransformation):
            #check whether feature is constant
            if test_unique_values == len(my_globale_module.preprocessed_folds_global):
                return None

            ## check if we computed an equivalent feature before
            materialized_all = []
            for fold_ii in range(len(my_globale_module.preprocessed_folds_global)):
                materialized_all.extend(test_transformed[fold_ii].flatten())
            materialized = tuple(materialized_all)
            if materialized in my_globale_module.materialized_set:
                return None

        if 'training_all' in list(candidate.parents)[0].runtime_properties:
            training_all_input = np.hstack(
                [p.runtime_properties['training_all'] for p in candidate.parents])
            one_test_set_transformed_input = np.hstack(
                [p.runtime_properties['one_test_set_transformed'] for p in candidate.parents])

            candidate.transformation.fit(training_all_input, my_globale_module.train_y_all_target_global)
            training_all = candidate.transformation.transform(training_all_input)
            one_test_set_transformed = candidate.transformation.transform(one_test_set_transformed_input)



    evaluated = True
    if  (
            (
                    isinstance(candidate.transformation, MinusTransformation) or
                    (isinstance(candidate.transformation, HigherOrderCommutativeTransformation) and candidate.transformation.method == np.nansum) or
                    (isinstance(candidate.transformation, NonCommutativeBinaryTransformation) and candidate.transformation.method == np.subtract)
            ) \
                and \
            (
              my_globale_module.classifier_global == LogisticRegression or
              my_globale_module.classifier_global == LinearRegression
            )
        ):
        candidate.runtime_properties['score'] = np.max([p.runtime_properties['score'] for p in candidate.parents])
        candidate.runtime_properties['test_score'] = -1.0
        candidate.runtime_properties['hyperparameters'] = None
        y_pred = None
        evaluated = False
        candidate.runtime_properties['passed'] = True
    else:
        candidate.runtime_properties['passed'] = False
        candidate.runtime_properties['score'], candidate.runtime_properties['test_score'], candidate.runtime_properties['hyperparameters'], test_fold_predictions, candidate.runtime_properties['fold_scores'], my_clf, candidate.runtime_properties['mean_scores'], candidate.runtime_properties['additional_metrics'] = grid_search(candidate, train_transformed, test_transformed, training_all, one_test_set_transformed,
            my_globale_module.grid_search_parameters_global, my_globale_module.score_global, my_globale_module.classifier_global, my_globale_module.target_train_folds_global, my_globale_module.target_test_folds_global, my_globale_module.train_y_all_target_global, my_globale_module.test_target_global)

        '''
        candidate.runtime_properties['score'], candidate.runtime_properties['test_score'], candidate.runtime_properties[
            'hyperparameters'], test_fold_predictions, candidate.runtime_properties['fold_scores'], my_clf, \
        candidate.runtime_properties['mean_scores'], candidate.runtime_properties['additional_metrics'] = hyperopt_search(
            train_transformed, test_transformed, training_all, one_test_set_transformed,
            my_globale_module.grid_search_parameters_global, my_globale_module.score_global,
            my_globale_module.classifier_global, my_globale_module.target_train_folds_global,
            my_globale_module.target_test_folds_global, my_globale_module.train_y_all_target_global,
            my_globale_module.test_target_global)
        '''



        #if True:
        #    candidate.runtime_properties['coef_'] = my_clf.coef_


        if Config.get_default('store.predictions', 'False') == 'True':
            candidate.runtime_properties['test_fold_predictions'] = test_fold_predictions


            '''
            ## check whether prediction already exists
            materialized_all = []
            for fold_ii in range(len(my_globale_module.preprocessed_folds_global)):
                materialized_all.extend(candidate.runtime_properties['test_fold_predictions'][fold_ii].flatten())
            materialized = tuple(materialized_all)
            if materialized in my_globale_module.predictions_set:
                return None
            '''




    if isinstance(candidate.transformation, OneHotTransformation) or isinstance(candidate, RawFeature) or not evaluated or ((candidate.runtime_properties['score'] - np.max([p.runtime_properties['score'] for p in candidate.parents])) / my_globale_module.complexity_delta_global) * my_globale_module.score_global._sign > my_globale_module.epsilon_global:
        candidate.runtime_properties['passed'] = True
        if not isinstance(candidate, RawFeature):
            candidate.runtime_properties['train_transformed'] = train_transformed
            candidate.runtime_properties['test_transformed'] = test_transformed

            if Config.get_default('score.test', 'False') == 'True' and type(training_all) != type(None):
                candidate.runtime_properties['training_all'] = training_all
                candidate.runtime_properties['one_test_set_transformed'] = one_test_set_transformed

            # derive properties
            if not isinstance(candidate, RawFeature):
                candidate.derive_properties(candidate.runtime_properties['train_transformed'][0])

                #avoid nan for specific ml models
                if candidate.properties['missing_values'] and my_globale_module.classifier_global == LogisticRegression:
                    return None


        # remove parents' materialization
        candidate.get_name()
        candidate.get_complexity()
        candidate.get_sympy_representation()
        if my_globale_module.remove_parents:
            candidate.parents = None
        return candidate


    return None


def evaluate_catch(candidate_id: int):
    result = None
    time_start_gs = time.time()
    try:
        result = evaluate(candidate_id)
    except Exception as e:
        warnings.warn(str(my_globale_module.candidate_list_global[candidate_id]) + " -> " + str(e), RuntimeWarning)

        '''
        candidate.runtime_properties['exception'] = e
        candidate.runtime_properties['score'] = -1.0
        candidate.runtime_properties['test_score'] = -1.0
        candidate.runtime_properties['hyperparameters'] = {}
        '''

    if type(result) != type(None):
        result.runtime_properties['execution_time'] = time.time() - time_start_gs
        result.runtime_properties['global_time'] = time.time() - my_globale_module.global_starting_time_global

    return result

def evaluate_no_catch(candidate_id: int):
    time_start_gs = time.time()

    result = evaluate(candidate_id)

    if result != None:
        result.runtime_properties['execution_time'] = time.time() - time_start_gs
        result.runtime_properties['global_time'] = time.time() - my_globale_module.global_starting_time_global

    return result

def evaluate_candidates_parallel(candidates: List[CandidateFeature], n_jobs: int = int(Config.get_default("parallelism", mp.cpu_count()))) -> List[CandidateFeature]:

    my_globale_module.candidate_list_global = candidates

    with mp.Pool(processes=n_jobs) as pool:
        my_function = evaluate_catch
        candidates_ids = list(range(len(candidates)))

        if Config.get_default("show_progess", 'True') == 'True':
            results = []
            for x in tqdm.tqdm(pool.imap_unordered(my_function, candidates_ids), total=len(candidates_ids)):
                results.append(x)
        else:
            results = pool.map(my_function, candidates_ids)


    return results