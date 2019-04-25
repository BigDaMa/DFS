from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import time
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.linear_model import LogisticRegression
import pickle
import multiprocessing as mp
from fastsklearnfeature.configuration.Config import Config
import itertools
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
import copy
from fastsklearnfeature.candidate_generation.feature_space.one_hot import get_transformation_for_cat_feature_space
from fastsklearnfeature.feature_selection.evaluation.CachedEvaluationFramework import CachedEvaluationFramework
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sympy
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.scorer import r2_scorer
from sklearn.metrics.scorer import neg_mean_squared_error_scorer
import fastsklearnfeature.feature_selection.evaluation.my_globale_module as my_globale_module
from fastsklearnfeature.feature_selection.evaluation.run_evaluation import evaluate_candidates


import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by MinMaxScaler.")
#warnings.filterwarnings("ignore", message="Data with input dtype object was converted to float64 by MinMaxScaler.")
#warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


class GlobalTraversalCognito(CachedEvaluationFramework):
    def __init__(self, dataset_config, classifier=LogisticRegression, grid_search_parameters={'penalty': ['l2'],
                                                                                                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'solver': ['lbfgs'],
                                                                                                'class_weight': ['balanced'],
                                                                                                'max_iter': [10000],
                                                                                                'multi_class':['auto']
                                                                                                },
                 transformation_producer=get_transformation_for_cat_feature_space,
                 epsilon=0.0,
                 c_max=2,
                 folds=10,
                 score=make_scorer(f1_score, average='micro'),
                 max_seconds=None,
                 save_logs=False,
                 lambda_threshold=2
                 ):
        super(GlobalTraversalCognito, self).__init__(dataset_config, classifier, grid_search_parameters,
                                                        transformation_producer)
        self.epsilon = -np.inf
        self.c_max = c_max
        self.folds = folds
        self.score = score
        self.save_logs = save_logs
        self.lambda_threshold=lambda_threshold

        self.max_timestamp = None
        if type(max_seconds) != type(None):
            self.max_timestamp = time.time() + max_seconds

        #https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    def partition(self, number):
        answer = set()
        answer.add((number,))
        for x in range(1, number):
            for y in self.partition(number - x):
                answer.add(tuple(sorted((x,) + y)))
        return answer

    def get_all_features_below_n_cost(self, cost):
        filtered_candidates = []
        for i in range(len(self.candidates)):
            if (self.candidates[i].get_number_of_transformations() + 1) <= cost:
                filtered_candidates.append(self.candidates[i])
        return filtered_candidates

    def get_all_features_equal_n_cost(self, cost, candidates):
        filtered_candidates = []
        for i in range(len(candidates)):
            if (candidates[i].get_number_of_transformations() + 1) == cost:
                filtered_candidates.append(candidates[i])
        return filtered_candidates



    def get_all_possible_representations_for_step_x(self, x, candidates):

        all_representations = set()
        partitions = self.partition(x)

        #get candidates of partitions
        candidates_with_cost_x = {}
        for i in range(x+1):
            candidates_with_cost_x[i] = self.get_all_features_equal_n_cost(i, candidates)

        for p in partitions:
            current_list = itertools.product(*[candidates_with_cost_x[pi] for pi in p])
            for c_output in current_list:
                if len(set(c_output)) == len(p):
                    all_representations.add(frozenset(c_output))

        return all_representations


    def filter_candidate(self, candidate):
        working_features: List[CandidateFeature] = []
        try:
            candidate.fit(self.dataset.splitted_values['train'])
            candidate.transform(self.dataset.splitted_values['train'])
            working_features.append(candidate)
        except:
            pass
        return working_features


    def filter_failing_in_parallel(self):
        pool = mp.Pool(processes=int(Config.get("parallelism")))
        results = pool.map(self.filter_candidate, self.candidates)
        return list(itertools.chain(*results))


    def generate_features(self, transformations: List[Transformation], features: List[CandidateFeature], all_evaluated_features: Set) -> List[CandidateFeature]:
        generated_features: List[CandidateFeature] = []
        for t_i in transformations:
            for f_i in t_i.get_combinations(features):
                if t_i.is_applicable(f_i):
                    sympy_representation = t_i.get_sympy_representation([p.get_sympy_representation() for p in f_i])
                    try:
                        if len(sympy_representation.free_symbols) > 0: # if expression is not constant
                            if not sympy_representation in all_evaluated_features:
                                candidate = CandidateFeature(copy.deepcopy(t_i), f_i)  # do we need a deep copy here?
                                candidate.sympy_representation = copy.deepcopy(sympy_representation)
                                generated_features.append(candidate)
                                all_evaluated_features.add(sympy_representation)
                            else:
                                #print("skipped: " + str(sympy_representation))
                                pass
                    except:
                        pass
        return generated_features


    def get_length_2_partition(self, cost: int) -> List[List[int]]:
        partition: List[List[int]] = []

        p = cost - 1
        while p >= cost - p:
            partition.append([p, cost - p])
            p = p - 1
        return partition

    #generate combinations for binary transformations
    def generate_merge(self, a: List[CandidateFeature], b: List[CandidateFeature], order_matters=False, repetition_allowed=False) -> List[List[CandidateFeature]]:
        # e.g. sum
        if not order_matters and repetition_allowed:
            return set([frozenset([x, y]) if x != y else (x, x) for x, y in itertools.product(*[a, b])])

        # feature concat, but does not work
        if not order_matters and not repetition_allowed:
            return set([frozenset([x, y]) for x, y in itertools.product(*[a, b]) if x != y])

        if order_matters and repetition_allowed:
            order = set(list(itertools.product(*[a, b])))
            order = order.union(set(list(itertools.product(*[b, a]))))
            return order

        # e.g. subtraction
        if order_matters and not repetition_allowed:
            order = set([(x, y) for x, y in itertools.product(*[a, b]) if x != y])
            order = order.union([(x, y) for x, y in itertools.product(*[b, a]) if x != y])
            return order







    def generate_merge_for_combination(self, all_evaluated_features, a: List[CandidateFeature], b: List[CandidateFeature]) -> Set[Set[CandidateFeature]]:
        cat_candidates_to_be_applied = []
        id_t = IdentityTransformation(None)
        for a_i in range(len(a)):
            for b_i in range(len(b)):
                combo = [a[a_i], b[b_i]]
                if id_t.is_applicable(combo):
                    sympy_representation = id_t.get_sympy_representation([p.get_sympy_representation() for p in combo])
                    if not sympy_representation in all_evaluated_features:
                        cat_candidate = CandidateFeature(copy.deepcopy(id_t), combo)
                        cat_candidate.sympy_representation = copy.deepcopy(sympy_representation)
                        all_evaluated_features.add(sympy_representation)
                        cat_candidates_to_be_applied.append(cat_candidate)

        return cat_candidates_to_be_applied


    # filter candidates that use one raw feature twice
    def filter_non_unique_combinations(self, candidates: List[CandidateFeature]):
        filtered_list: List[CandidateFeature] = []
        for candidate in candidates:
            all_raw_features = candidate.get_raw_attributes()
            if len(all_raw_features) == len(set(all_raw_features)):
                filtered_list.append(candidate)
        return filtered_list


    def materialize_raw_features(self, candidate):
        train_transformed = [None] * len(self.preprocessed_folds)
        test_transformed = [None] * len(self.preprocessed_folds)

        # test
        training_all = None
        one_test_set_transformed = None

        candidate.fit(self.dataset.splitted_values['train'])
        raw_feature = candidate.transform(self.dataset.splitted_values['train'])

        for fold in range(len(self.preprocessed_folds)):
            train_transformed[fold] = raw_feature[self.preprocessed_folds[fold][0]]
            test_transformed[fold] = raw_feature[self.preprocessed_folds[fold][1]]

        if Config.get_default('score.test', 'False') == 'True':
            training_all = raw_feature
            if Config.get_default('instance.selection', 'False') == 'True':
                candidate.fit(self.train_X_all)
                training_all = candidate.transform(self.train_X_all)
            one_test_set_transformed = candidate.transform(self.dataset.splitted_values['test'])

            candidate.runtime_properties['training_all'] = training_all
            candidate.runtime_properties['one_test_set_transformed'] = one_test_set_transformed

        candidate.runtime_properties['train_transformed'] = train_transformed
        candidate.runtime_properties['test_transformed'] = test_transformed

    def count_smaller_or_equal(self, candidates: List[CandidateFeature], current_score):
        count_smaller_or_equal = 0
        for c in candidates:
            if c.runtime_properties['score'] <= current_score:
                count_smaller_or_equal += 1
        return count_smaller_or_equal

    # P(Accuracy <= current) -> 1.0 = highest accuracy
    def getAccuracyScore(self, current_score, complexity, cost_2_raw_features, cost_2_unary_transformed, cost_2_binary_transformed, cost_2_combination):
        count_smaller_or_equal_v = 0
        count_all = 0
        for c in range(1, complexity + 1):
            if c in cost_2_raw_features:
                count_smaller_or_equal_v += self.count_smaller_or_equal(cost_2_raw_features[c], current_score)
            if c in cost_2_unary_transformed:
                count_smaller_or_equal_v += self.count_smaller_or_equal(cost_2_unary_transformed[c], current_score)
            if c in cost_2_binary_transformed:
                count_smaller_or_equal_v += self.count_smaller_or_equal(cost_2_binary_transformed[c], current_score)
            if c in cost_2_combination:
                count_smaller_or_equal_v += self.count_smaller_or_equal(cost_2_combination[c], current_score)

            if c in cost_2_raw_features:
                count_all += len(cost_2_raw_features[c])
            if c in cost_2_unary_transformed:
                count_all += len(cost_2_unary_transformed[c])
            if c in cost_2_binary_transformed:
                count_all += len(cost_2_binary_transformed[c])
            if c in cost_2_combination:
                count_all += len(cost_2_combination[c])

        return count_smaller_or_equal_v / float(count_all)

    # P(Complexity >= current) -> 1.0 = lowest complexity
    def getSimplicityScore(self, current_complexity, complexity, cost_2_raw_features, cost_2_unary_transformed, cost_2_binary_transformed, cost_2_combination):
        count_greater_or_equal_v = 0
        count_all = 0

        for c in range(1, complexity + 1):
            if c >= current_complexity:
                if c in cost_2_raw_features:
                    count_greater_or_equal_v += len(cost_2_raw_features[c])
                if c in cost_2_unary_transformed:
                    count_greater_or_equal_v += len(cost_2_unary_transformed[c])
                if c in cost_2_binary_transformed:
                    count_greater_or_equal_v += len(cost_2_binary_transformed[c])
                if c in cost_2_combination:
                    count_greater_or_equal_v += len(cost_2_combination[c])

            if c in cost_2_raw_features:
                count_all += len(cost_2_raw_features[c])
            if c in cost_2_unary_transformed:
                count_all += len(cost_2_unary_transformed[c])
            if c in cost_2_binary_transformed:
                count_all += len(cost_2_binary_transformed[c])
            if c in cost_2_combination:
                count_all += len(cost_2_combination[c])

        return count_greater_or_equal_v / float(count_all)

    def harmonic_mean(self, complexity, accuracy):
        return (2 * complexity * accuracy) / (complexity + accuracy)


    def run(self):

        self.global_starting_time = time.time()

        # generate all candidates
        self.generate()
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        unary_transformations, binary_transformations = self.transformation_producer(self.train_X_all, self.raw_features)



        cost_2_raw_features: Dict[int, List[CandidateFeature]] = {}
        cost_2_unary_transformed: Dict[int, List[CandidateFeature]] = {}
        cost_2_binary_transformed: Dict[int, List[CandidateFeature]] = {}
        cost_2_combination: Dict[int, List[CandidateFeature]] = {}

        if self.save_logs:
            cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = {}

        self.complexity_delta = 1.0

        max_feature = CandidateFeature(IdentityTransformation(None), [self.raw_features[0]])
        max_feature.runtime_properties['score'] = -float("inf")

        all_evaluated_features = set()

        my_globale_module.global_starting_time_global = copy.deepcopy(self.global_starting_time)
        my_globale_module.grid_search_parameters_global = copy.deepcopy(self.grid_search_parameters)
        my_globale_module.score_global = copy.deepcopy(self.score)
        my_globale_module.classifier_global = copy.deepcopy(self.classifier)
        my_globale_module.target_train_folds_global = copy.deepcopy(self.target_train_folds)
        my_globale_module.target_test_folds_global = copy.deepcopy(self.target_test_folds)
        my_globale_module.train_y_all_target_global = copy.deepcopy(self.train_y_all_target)
        my_globale_module.test_target_global = copy.deepcopy(self.test_target)
        my_globale_module.max_timestamp_global = copy.deepcopy(self.max_timestamp)
        my_globale_module.preprocessed_folds_global = copy.deepcopy(self.preprocessed_folds)
        my_globale_module.epsilon_global = copy.deepcopy(self.epsilon)
        my_globale_module.complexity_delta_global = copy.deepcopy(self.complexity_delta)




        ############################

        # start

        ############################

        current_layer = []
        c = 1

        cost_2_raw_features[c]: List[CandidateFeature] = []
        # print(self.raw_features)
        for raw_f in self.raw_features:
            sympy_representation = sympy.Symbol('X' + str(raw_f.column_id))
            raw_f.sympy_representation = sympy_representation
            all_evaluated_features.add(sympy_representation)
            if raw_f.is_numeric():
                current_layer.append(raw_f)
                # print("numeric: " + str(raw_f))
            else:
                raw_f.runtime_properties['score'] = 0.0
                cost_2_raw_features[c].append(raw_f)
                # print("nonnumeric: " + str(raw_f))

            self.materialize_raw_features(raw_f)
            raw_f.derive_properties(raw_f.runtime_properties['train_transformed'][0])

        # now evaluate all from this layer
        # print(current_layer)
        print("----------- Evaluation of " + str(len(current_layer)) + " representations -----------")
        results = evaluate_candidates(current_layer)
        print("----------- Evaluation Finished -----------")

        layer_end_time = time.time() - self.global_starting_time

        # calculate whether we drop the evaluated candidate
        for candidate in results:
            if type(candidate) != type(None):
                candidate.runtime_properties['layer_end_time'] = layer_end_time

                # print(str(candidate) + " -> " + str(candidate.runtime_properties['score']))

                if candidate.runtime_properties['score'] > max_feature.runtime_properties['score']:
                    max_feature = candidate

                if candidate.runtime_properties['passed']:
                    if isinstance(candidate, RawFeature):
                        if not c in cost_2_raw_features:
                            cost_2_raw_features[c]: List[CandidateFeature] = []
                        cost_2_raw_features[c].append(candidate)
                    elif isinstance(candidate.transformation, UnaryTransformation):
                        if not c in cost_2_unary_transformed:
                            cost_2_unary_transformed[c]: List[CandidateFeature] = []
                        cost_2_unary_transformed[c].append(candidate)
                    elif isinstance(candidate.transformation, IdentityTransformation):
                        if not c in cost_2_combination:
                            cost_2_combination[c]: List[CandidateFeature] = []
                        cost_2_combination[c].append(candidate)
                    else:
                        if not c in cost_2_binary_transformed:
                            cost_2_binary_transformed[c]: List[CandidateFeature] = []
                        cost_2_binary_transformed[c].append(candidate)
                else:
                    if self.save_logs:
                        if not c in cost_2_dropped_evaluated_candidates:
                            cost_2_dropped_evaluated_candidates[c]: List[CandidateFeature] = []
                        cost_2_dropped_evaluated_candidates[c].append(candidate)

        print(cost_2_raw_features[c])

        #select next representation

        next_id = np.argmax([rf.runtime_properties['score'] for rf in cost_2_raw_features[1]])
        next_rep = cost_2_raw_features[c][next_id]

        max_rep = next_rep

        number_runs= 200

        all_representations = []
        all_representations.extend(cost_2_raw_features[c])

        for runs in range(number_runs):
            #get next

            current_max_rep = all_representations[0]
            for rep in all_representations:
                if rep.runtime_properties['score'] > current_max_rep.runtime_properties['score']:
                    current_max_rep = rep
            all_representations.remove(current_max_rep)
            next_rep = current_max_rep

            if max_rep.runtime_properties['score'] < current_max_rep.runtime_properties['score']:
                max_rep = current_max_rep
                print("max representation: " + str(max_rep))

            if 'test_score' in next_rep.runtime_properties:
                print(str(next_rep) + " cv score: " + str(next_rep.runtime_properties['score']) + " test: " + str(
                    next_rep.runtime_properties['test_score']))
            else:
                print(str(next_rep))


                #######################
            #create branch
            #######################
            current_layer = []
            # first unary
            if not isinstance(next_rep.transformation, IdentityTransformation):
                current_layer.extend(self.generate_features(unary_transformations, [next_rep], all_evaluated_features))

            # second binary
            if not isinstance(next_rep.transformation, IdentityTransformation):
                binary_candidates_to_be_applied = []
                for bt in binary_transformations:
                    list_of_combinations = self.generate_merge([next_rep], cost_2_raw_features[1],
                                                               bt.parent_feature_order_matters,
                                                               bt.parent_feature_repetition_is_allowed)
                    # print(list_of_combinations)
                    for combo in list_of_combinations:
                        if bt.is_applicable(combo):
                            sympy_representation = bt.get_sympy_representation(
                                [p.get_sympy_representation() for p in combo])
                            try:
                                if len(sympy_representation.free_symbols) > 0:  # if expression is not constant
                                    if not sympy_representation in all_evaluated_features:
                                        bin_candidate = CandidateFeature(copy.deepcopy(bt), combo)
                                        bin_candidate.sympy_representation = copy.deepcopy(sympy_representation)
                                        binary_candidates_to_be_applied.append(bin_candidate)
                                        all_evaluated_features.add(sympy_representation)
                                    else:
                                        # print(str(bin_candidate) + " skipped: " + str(sympy_representation))
                                        pass
                                else:
                                    # print(str(bin_candidate) + " skipped: " + str(sympy_representation))
                                    pass
                            except:
                                pass
                current_layer.extend(binary_candidates_to_be_applied)

            # third: feature combinations
            '''
            combinations_to_be_applied = self.generate_merge_for_combination(all_evaluated_features, [next_rep], cost_2_raw_features[1])
            current_layer.extend(combinations_to_be_applied)
            '''

            #print(current_layer)

            # select next representation
            new_representations = evaluate_candidates(current_layer)
            #print(new_representations)

            for rep in new_representations:
                if rep != None:
                    all_representations.append(rep)



if __name__ == '__main__':
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)

    #dataset = (Config.get('data_path') + "/phpn1jVwe_mammography.csv", 6)
    #dataset = (Config.get('data_path') + "/dataset_23_cmc_contraceptive.csv", 9)
    #dataset = (Config.get('data_path') + "/dataset_31_credit-g_german_credit.csv", 20)
    #dataset = (Config.get('data_path') + '/dataset_53_heart-statlog_heart.csv', 13)
    #dataset = (Config.get('data_path') + '/ILPD.csv', 10)
    #dataset = (Config.get('data_path') + '/iris.data', 4)
    #dataset = (Config.get('data_path') + '/data_banknote_authentication.txt', 4)
    #dataset = (Config.get('data_path') + '/ecoli.data', 8)
    #dataset = (Config.get('data_path') + '/breast-cancer.data', 0)
    #dataset = (Config.get('data_path') + '/transfusion.data', 4)
    #dataset = (Config.get('data_path') + '/test_categorical.data', 4)
    #dataset = ('../configuration/resources/data/transfusion.data', 4)
    #dataset = (Config.get('data_path') + '/wine.data', 0)

    dataset = (Config.get('data_path') + '/house_price.csv', 79)
    #dataset = (Config.get('data_path') + '/synthetic_data.csv', 3)





    start = time.time()



    #regression
    selector = GlobalTraversalCognito(dataset,classifier=LinearRegression,grid_search_parameters={'fit_intercept': [True, False],'normalize': [True, False]},score=r2_scorer,save_logs=True)

    #selector = ComplexityDrivenFeatureConstruction(dataset, classifier=LinearRegression, grid_search_parameters={'fit_intercept': [True, False],'normalize': [True, False]}, score=neg_mean_squared_error_scorer, c_max=5, save_logs=True)

    #classification
    #selector = GlobalTraversalCognito(dataset, c_max=3, folds=10, max_seconds=None, save_logs=True)

    #selector = ComplexityDrivenFeatureConstruction(dataset, c_max=5, folds=10,
    #                                               max_seconds=None, save_logs=True, transformation_producer=get_transformation_for_cat_feature_space)


    '''
    selector = ComplexityDrivenFeatureConstruction(dataset,
                                                   classifier=KNeighborsClassifier,
                                                   grid_search_parameters={'n_neighbors': np.arange(3, 10),
                                                                           'weights': ['uniform', 'distance'],
                                                                           'metric': ['minkowski', 'euclidean',
                                                                                      'manhattan']},
                                                   c_max=5, save_logs=True) #,transformation_producer=get_transformation_for_cat_feature_space)
    '''

    selector.run()

    print(time.time() - start)







