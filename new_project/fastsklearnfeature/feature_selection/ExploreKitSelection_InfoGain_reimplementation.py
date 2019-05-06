from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import Set
import time
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
from fastsklearnfeature.configuration.Config import Config
import itertools
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
import copy
from fastsklearnfeature.candidate_generation.feature_space.explorekit_transformations import get_transformation_for_feature_space
from typing import List
import numpy as np
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastsklearnfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import GroupByThenGenerator
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
import pickle
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.scorer import r2_scorer
from sklearn.metrics.scorer import neg_mean_squared_error_scorer

from sklearn.feature_selection import mutual_info_classif
from fastsklearnfeature.feature_selection.evaluation.EvaluationFramework import EvaluationFramework

import warnings
warnings.filterwarnings("ignore")

class ExploreKitSelection_iterative_search(EvaluationFramework):
    def __init__(self, dataset_config, classifier=LogisticRegression, grid_search_parameters={'classifier__penalty': ['l2'],
                                                                                                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'classifier__solver': ['lbfgs'],
                                                                                                'classifier__class_weight': ['balanced'],
                                                                                                'classifier__max_iter': [10000],
                                                                                                'classifier__multi_class':['auto']
                                                                                                },
                 transformation_producer=get_transformation_for_feature_space,
                 reader=None,
                 score=make_scorer(f1_score, average='micro'),
                 folds=10
                 ):
        self.dataset_config = dataset_config
        self.classifier = classifier
        self.grid_search_parameters = grid_search_parameters
        self.transformation_producer = transformation_producer
        self.reader = reader
        self.score = score
        self.folds = folds





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
            return list(itertools.product(*[a, b]))

        # feature concat, but does not work
        if not order_matters and not repetition_allowed:
            return [[x, y] for x, y in itertools.product(*[a, b]) if x != y]

        if order_matters and repetition_allowed:
            order = set(list(itertools.product(*[a, b])))
            order = order.union(set(list(itertools.product(*[b, a]))))
            return list(order)

        # e.g. subtraction
        if order_matters and not repetition_allowed:
            order = [[x, y] for x, y in itertools.product(*[a, b]) if x != y]
            order.extend([[x, y] for x, y in itertools.product(*[b, a]) if x != y])
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

    def generate_features1(self, transformations: List[Transformation], features: List[CandidateFeature]):
        generated_features: List[CandidateFeature] = []
        for t_i in transformations:
            for f_i in t_i.get_combinations(features):
                if t_i.is_applicable(f_i):
                    can = CandidateFeature(copy.deepcopy(t_i), f_i)
                    can.properties['type']= 'float'
                    generated_features.append(can) # do we need a deep copy here?
                    #if output is multidimensional adapt here
        return generated_features


    def produce_features(self):
        unary_transformations: List[UnaryTransformation] = []
        unary_transformations.append(PandasDiscretizerTransformation(number_bins=10))
        unary_transformations.append(MinMaxScalingTransformation())

        higher_order_transformations: List[Transformation] = []
        higher_order_transformations.extend(
            HigherOrderCommutativeClassGenerator(2, methods=[np.nansum, np.nanprod]).produce())
        higher_order_transformations.extend(NumpyBinaryClassGenerator(methods=[np.divide, np.subtract]).produce())

        # count is missing
        higher_order_transformations.extend(GroupByThenGenerator(2, methods=[np.nanmax,
                                                                             np.nanmin,
                                                                             np.nanmean,
                                                                             np.nanstd]).produce())

        Fui = self.generate_features1(unary_transformations, self.raw_features)

        Fi_and_Fui = []
        Fi_and_Fui.extend(self.raw_features)
        Fi_and_Fui.extend(Fui)

        Foi = self.generate_features1(higher_order_transformations, Fi_and_Fui)

        Foui = self.generate_features1(unary_transformations, Foi)

        Fi_cand = []
        Fi_cand.extend(Fui)
        Fi_cand.extend(Foi)
        Fi_cand.extend(Foui)

        return Fi_cand


    def get_info_gain_of_feature(self, candidate: CandidateFeature):
        try:
            new_candidate = CandidateFeature(IdentityTransformation(2), [self.base_features, candidate])
            X = new_candidate.pipeline.fit_transform(self.dataset.splitted_values['train'], self.train_y_all_target)
            return mutual_info_classif(X, self.train_y_all_target)[-1]
        except:
            return 0.0

    def evaluate_ranking(self, candidates):
        self.preprocessed_folds = []
        pool = mp.Pool(processes=int(Config.get("parallelism")))
        results = pool.map(self.get_info_gain_of_feature, candidates)
        return results

    def calculate_complexity(self, feature_set: List[CandidateFeature]):
        complexity = 0
        for f in feature_set:
            complexity += f.get_complexity()
        return complexity


    def run(self):

        self.global_starting_time = time.time()

        # generate all candidates
        self.generate()

        for raw_f in self.raw_features:
            raw_f.properties['type'] = 'float'



        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        myfolds = copy.deepcopy(list(self.preprocessed_folds))

        R_w = 15000
        max_iterations = 15 #15
        threshold_f = 0.001
        epsilon_w = 0.01
        threshold_w = 0.0

        all_features = self.produce_features()

        print(len(all_features))

        self.base_features = CandidateFeature(IdentityTransformation(len(self.raw_features)), self.raw_features)

        results = {}

        for i in range(max_iterations):

            print("base features: " + str(self.base_features))

            results[i] = self.evaluate_candidates([self.base_features], myfolds)[0]
            print(results[i])
            print(results[i].runtime_properties)

            feature_scores = self.evaluate_ranking(all_features)
            ids = np.argsort(np.array(feature_scores) * -1)
            print(feature_scores)

            best_improvement_so_far = np.NINF
            best_Feature_So_Far = None
            evaluated_candidate_features = 0
            for f_i in range(len(feature_scores)):
                if feature_scores[ids[f_i]] < threshold_f:
                    break

                current_feature_set = CandidateFeature(IdentityTransformation(2), [self.base_features, all_features[ids[f_i]]])
                print(current_feature_set)
                result = self.evaluate_candidates([current_feature_set], myfolds)[0]
                evaluated_candidate_features += 1
                improvement = result.runtime_properties['score'] - results[i].runtime_properties['score']

                print("Candidate: " + str(all_features[ids[f_i]]) + " score: " + str(result.runtime_properties['score']) + " info: " + str(feature_scores[ids[f_i]]))
                print("improvement: " + str(improvement))
                if improvement > best_improvement_so_far:
                    best_improvement_so_far = improvement
                    best_Feature_So_Far = result

                    results[i] = best_Feature_So_Far
                    results[i].runtime_properties['score_improvement'] = improvement
                    results[i].runtime_properties['info_gain'] = feature_scores[ids[f_i]]

                    pickle.dump(results, open(Config.get("tmp.folder") + "/explorekit_results.p", "wb"))

                if improvement >= epsilon_w:
                    break
                if evaluated_candidate_features >= R_w:
                    break

            if best_improvement_so_far > threshold_w:
                self.base_features = best_Feature_So_Far
            else:
                return self.base_features

            all_features_new = []
            for i in range(len(feature_scores)):
                if feature_scores[i] >= 0:
                    all_features_new.append(all_features[i])
            all_features = all_features_new
        return results









#statlog_heart.csv=/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv
#statlog_heart.target=13

if __name__ == '__main__':
    #dataset = (Config.get('statlog_heart.csv'), 13)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6)


    #dataset = (Config.get('iris.csv'), 4)
    #dataset = (Config.get('banknote.csv'), 4)
    #dataset = (Config.get('ecoli.csv'), 8)
    #dataset = (Config.get('abalone.csv'), 8)
    #dataset = (Config.get('breastcancer.csv'), 0)
    dataset = (Config.get('data_path') + '/transfusion.data', 4)

    from fastsklearnfeature.reader.OnlineOpenMLReader import OnlineOpenMLReader

    from fastsklearnfeature.feature_selection.evaluation.openMLdict import openMLname2task

    #dataset = None
    #task_id = openMLname2task['transfusion']

    #selector = ExploreKitSelection_iterative_search(dataset, reader=OnlineOpenMLReader(task_id))
    selector = ExploreKitSelection_iterative_search(dataset)
    #selector = ExploreKitSelection(dataset, KNeighborsClassifier(), {'n_neighbors': np.arange(3,10), 'weights': ['uniform','distance'], 'metric': ['minkowski','euclidean','manhattan']})

    selector.run()







