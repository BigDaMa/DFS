from typing import List, Dict, Set
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from fastsklearnfeature.configuration.Config import Config
from fastsklearnfeature.candidate_generation.feature_space.explorekit_transformations import get_transformation_for_feature_space
from fastsklearnfeature.feature_selection.evaluation.EvaluationFramework import EvaluationFramework
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import time
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import itertools
from collections import deque

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class NestedCachedEvaluationFramework(EvaluationFramework):
    def __init__(self, dataset_config, classifier=LogisticRegression, grid_search_parameters={'penalty': ['l2'],
                                                                                                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'solver': ['lbfgs'],
                                                                                                'class_weight': ['balanced'],
                                                                                                'max_iter': [10000],
                                                                                                'multi_class':['auto']
                                                                                                },
                 transformation_producer=get_transformation_for_feature_space
                 ):
        super(NestedCachedEvaluationFramework, self).__init__(dataset_config, classifier, grid_search_parameters, transformation_producer)


    def generate_target(self):
        current_target = self.dataset.splitted_target['train']

        label_encoder = LabelEncoder()
        label_encoder.fit(current_target)

        self.current_target = label_encoder.transform(current_target)

        if Config.get_default('score.test', 'False') == 'True':
            self.test_target = label_encoder.transform(self.dataset.splitted_target['test'])
            self.train_y_all_target = label_encoder.transform(self.train_y_all)


        self.preprocessed_folds = []
        for train, test in StratifiedKFold(n_splits=self.folds, random_state=42).split(self.dataset.splitted_values['train'],
                                                                                       self.current_target):
            self.preprocessed_folds.append((train, test))






    def nested_grid_search(self, train_transformed, validation_transformed, train_and_validation_transformed, test_transformed, training_all, one_test_set_transformed):

        q = deque(range(self.folds))

        nested_test_score = []

        for outer_i in range(self.folds):
            test_fold = q.pop()

            hyperparam_to_score_list = {}
            my_keys = list(self.grid_search_parameters.keys())
            for parameter_combination in itertools.product(*[self.grid_search_parameters[k] for k in my_keys]):
                parameter_set = hashabledict(zip(my_keys, parameter_combination))
                hyperparam_to_score_list[parameter_set] = []

                for inner_i in range(self.folds - 1):
                    q.rotate(-1)
                    validation_fold = q[0]
                    training_folds = [q[training_i] for training_i in range(1, self.folds - 1)]

                    ################################################################
                    clf = self.classifier(**parameter_set)
                    clf.fit(train_transformed[frozenset(training_folds)], self.store_cv_sets_train_target[frozenset(training_folds)])
                    y_val_pred = clf.predict(validation_transformed[frozenset(training_folds)]) #validation
                    hyperparam_to_score_list[parameter_set].append(
                        f1_score(self.store_cv_sets_validation_target[frozenset(training_folds)], y_val_pred, average='micro'))


                    ################################################################

            #find best parameter
            best_param = None
            best_mean_cross_val_score = -1
            for parameter_config, score_list in hyperparam_to_score_list.items():
                mean_score = np.mean(score_list)
                if mean_score > best_mean_cross_val_score:
                    best_param = parameter_config
                    best_mean_cross_val_score = mean_score

            ######################

            clf = self.classifier(**best_param)
            clf.fit(train_and_validation_transformed[test_fold], self.store_cv_sets_train_and_validation_target[test_fold])
            y_test_pred = clf.predict(test_transformed[test_fold])

            nested_test_score.append(f1_score(self.store_cv_sets_test_target[test_fold], y_test_pred, average='micro'))



            ######################


            q.appendleft(test_fold)



        nested_cross_val_score = np.mean(nested_test_score)


        test_score = None
        if Config.get_default('score.test', 'False') == 'True':
            # refit to entire training and test on test set
            clf = self.classifier(**best_param)
            clf.fit(training_all, self.train_y_all_target)
            y_pred = clf.predict(one_test_set_transformed)

            test_score = f1_score(self.test_target, y_pred, average='micro')

            #np.save('/tmp/true_predictions', self.test_target)


        return nested_cross_val_score, test_score, best_param, y_pred





    def evaluate(self, candidate: CandidateFeature, score=make_scorer(f1_score, average='micro')):

        if type(self.max_timestamp) != type(None) and time.time() >= self.max_timestamp:
            raise RuntimeError('Out of time!')

        result = {}
        train_transformed = {}
        validation_transformed = {}
        train_and_validation_transformed = [None] * self.folds
        test_transformed = [None] * self.folds

        #test
        training_all = None
        one_test_set_transformed = None

        result['train_transformed'] = None
        result['validation_transformed'] = None
        result['train_and_validation_transformed'] = None
        result['test_transformed'] = None
        result['one_test_set_transformed'] = None

        if isinstance(candidate, RawFeature):

            if Config.get_default('score.test', 'False') == 'True':
                result['training_all'] = training_all = self.name_to_training_all[str(candidate)]
                result['one_test_set_transformed'] = one_test_set_transformed = self.name_to_one_test_set_transformed[str(candidate)]

            result['train_transformed'] = train_transformed = self.name_to_train_transformed[str(candidate)]
            result['validation_transformed'] = validation_transformed = self.name_to_validation_transformed[str(candidate)]
            result['train_and_validation_transformed'] = train_and_validation_transformed =self.name_to_train_and_validation_transformed[str(candidate)]
            result['test_transformed'] = test_transformed = self.name_to_test_transformed[str(candidate)]

        else:

            #print(self.name_to_train_transformed.keys())

            #merge columns from parents
            for key, value in self.name_to_train_transformed[str(list(candidate.parents)[0])].items():
                train_transformed_input = np.hstack([self.name_to_train_transformed[str(p)][key] for p in candidate.parents])
                validation_transformed_input = np.hstack([self.name_to_validation_transformed[str(p)][key] for p in candidate.parents])

                candidate.transformation.fit(train_transformed_input)

                train_transformed[key] = candidate.transformation.transform(train_transformed_input)
                validation_transformed[key] = candidate.transformation.transform(validation_transformed_input)

            for fold_i in range(self.folds):
                train_and_validation_transformed_input = np.hstack([self.name_to_train_and_validation_transformed[str(p)][fold_i] for p in candidate.parents])
                test_transformed_input = np.hstack([self.name_to_test_transformed[str(p)][fold_i] for p in candidate.parents])

                candidate.transformation.fit(train_and_validation_transformed_input)

                train_and_validation_transformed[fold_i] = candidate.transformation.transform(train_and_validation_transformed_input)
                test_transformed[fold_i] = candidate.transformation.transform(test_transformed_input)

            if Config.get_default('score.test', 'False') == 'True':
                training_all_input = np.hstack(
                    [self.name_to_training_all[str(p)] for p in candidate.parents])
                one_test_set_transformed_input = np.hstack(
                    [self.name_to_one_test_set_transformed[str(p)] for p in candidate.parents])

                candidate.transformation.fit(training_all_input)
                training_all = candidate.transformation.transform(training_all_input)
                one_test_set_transformed = candidate.transformation.transform(one_test_set_transformed_input)

        candidate.runtime_properties['score'], candidate.runtime_properties['test_score'], candidate.runtime_properties['hyperparameters'], y_pred = self.nested_grid_search(train_transformed, validation_transformed, train_and_validation_transformed, test_transformed, training_all, one_test_set_transformed)

        if Config.get_default('store.predictions', 'False') == 'True':
            candidate.runtime_properties['predictions'] = y_pred

        if not isinstance(candidate, RawFeature):
            #only save the transformed data if we need it in the future
            max_parent = np.max([p.runtime_properties['score'] for p in candidate.parents])
            accuracy_delta = candidate.runtime_properties['score'] - max_parent
            if accuracy_delta / self.complexity_delta > self.epsilon:

                result['train_transformed'] = train_transformed
                result['validation_transformed'] = validation_transformed
                result['train_and_validation_transformed'] = train_and_validation_transformed
                result['test_transformed'] = test_transformed


                result['training_all'] = training_all
                result['one_test_set_transformed'] = one_test_set_transformed

                # derive properties
                if not isinstance(candidate, RawFeature):
                    candidate.derive_properties(result['train_and_validation_transformed'][0])


        return result


    '''
    def evaluate_single_candidate(self, candidate):
        result = {}
        time_start_gs = time.time()
        result = self.evaluate(candidate)
        #print("feature: " + str(candidate) + " -> " + str(new_score))

        result['candidate'] = candidate
        result['execution_time'] = time.time() - time_start_gs
        result['global_time'] = time.time() - self.global_starting_time
        return result
    '''
















