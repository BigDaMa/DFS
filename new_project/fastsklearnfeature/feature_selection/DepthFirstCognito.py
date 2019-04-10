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
from fastsklearnfeature.candidate_generation.feature_space.explorekit_transformations import get_transformation_for_feature_space
from fastsklearnfeature.feature_selection.evaluation.CachedEvaluationFramework import CachedEvaluationFramework
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sympy

import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by MinMaxScaler.")
#warnings.filterwarnings("ignore", message="Data with input dtype object was converted to float64 by MinMaxScaler.")
#warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


class DepthFirstCognito(CachedEvaluationFramework):
    def __init__(self, dataset_config, classifier=LogisticRegression, grid_search_parameters={'penalty': ['l2'],
                                                                                                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'solver': ['lbfgs'],
                                                                                                'class_weight': ['balanced'],
                                                                                                'max_iter': [10000],
                                                                                                'multi_class':['auto']
                                                                                                },
                 transformation_producer=get_transformation_for_feature_space,
                 epsilon=0.0,
                 c_max=2,
                 folds=10,
                 max_seconds=None,
                 save_logs=False
                 ):
        super(DepthFirstCognito, self).__init__(dataset_config, classifier, grid_search_parameters,
                                                        transformation_producer)
        self.epsilon = epsilon
        self.c_max = c_max
        self.folds = folds
        self.save_logs = save_logs

        self.name_to_train_transformed = {}
        self.name_to_test_transformed = {}
        self.name_to_training_all = {}
        self.name_to_one_test_set_transformed = {}

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
                    candidate = CandidateFeature(copy.deepcopy(t_i), f_i) # do we need a deep copy here?
                    sympy_representation = candidate.get_sympy_representation()
                    if len(sympy_representation.free_symbols) > 0: # if expression is not constant
                        if not sympy_representation in all_evaluated_features:
                            all_evaluated_features.add(sympy_representation)
                            generated_features.append(candidate)
                        else:
                            print("skipped: " + str(sympy_representation))
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




    def get_features_from_identity_candidate(self, identity: CandidateFeature):
        my_list = set()
        if not isinstance(identity.transformation, IdentityTransformation):
            return set([str(identity)])

        for p in identity.parents:
            if not isinstance(p.transformation, IdentityTransformation):
                my_list.add(str(p))
            else:
                my_list = my_list.union(self.get_features_from_identity_candidate(p))
        return my_list


    def generate_merge_for_combination(self, a: List[CandidateFeature], b: List[CandidateFeature]) -> Set[Set[CandidateFeature]]:
        # feature concat, but does not work
        #if not order_matters and not repetition_allowed:
        #    return [[x, y] for x, y in itertools.product(*[a, b]) if x != y]
        result_list: Set[Set[CandidateFeature]] = set()

        for a_i in range(len(a)):
            for b_i in range(len(b)):
                #we have to check whether they intersect or not
                #so we climb down the transformation pipeline and gather all concatenated features
                set_a = self.get_features_from_identity_candidate(a[a_i])
                set_b = self.get_features_from_identity_candidate(b[b_i])
                if len(set_a.intersection(set_b)) == 0:
                    result_list.add(frozenset([a[a_i], b[b_i]]))

        return result_list


    # filter candidates that use one raw feature twice
    def filter_non_unique_combinations(self, candidates: List[CandidateFeature]):
        filtered_list: List[CandidateFeature] = []
        for candidate in candidates:
            all_raw_features = candidate.get_raw_attributes()
            if len(all_raw_features) == len(set(all_raw_features)):
                filtered_list.append(candidate)
        return filtered_list




    def run(self):

        self.global_starting_time = time.time()

        # generate all candidates
        self.generate()
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        unary_transformations, binary_transformations = self.transformation_producer()



        cost_2_raw_features: Dict[int, List[CandidateFeature]] = {}
        cost_2_unary_transformed: Dict[int, List[CandidateFeature]] = {}
        cost_2_binary_transformed: Dict[int, List[CandidateFeature]] = {}
        cost_2_combination: Dict[int, List[CandidateFeature]] = {}

        cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = {}

        self.complexity_delta = 1.0

        limit_runs = self.c_max + 1  # 5
        unique_raw_combinations = False


        baseline_score = 0.0#self.evaluate_candidates([CandidateFeature(DummyOneTransformation(None), [self.raw_features[0]])])[0]['score']
        #print("baseline: " + str(baseline_score))


        max_feature = CandidateFeature(IdentityTransformation(None), [self.raw_features[0]])
        max_feature.runtime_properties['score'] = -2

        all_evaluated_features = set()

        starting_feature = self.raw_features[np.random.randint(len(self.raw_features))]

        #apply all possible transformations





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
    dataset = (Config.get('data_path') + '/transfusion.data', 4)
    #dataset = (Config.get('data_path') + '/test_categorical.data', 4)
    #dataset = ('../configuration/resources/data/transfusion.data', 4)

    start = time.time()

    selector = DepthFirstCognito(dataset, c_max=5, folds=10, max_seconds=None, save_logs=True)


    '''
    selector = ComplexityDrivenFeatureConstruction(dataset,
                                         classifier=KNeighborsClassifier,
                                         grid_search_parameters={'n_neighbors': np.arange(3,10), 'weights': ['uniform','distance'], 'metric': ['minkowski','euclidean','manhattan']},
                                         c_max=3, save_logs=True)
    '''



    selector.run()

    print(time.time() - start)







