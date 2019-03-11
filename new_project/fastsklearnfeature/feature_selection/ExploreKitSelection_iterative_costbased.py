from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import numpy as np
from fastsklearnfeature.reader.Reader import Reader
from fastsklearnfeature.splitting.Splitter import Splitter
import time
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from fastsklearnfeature.configuration.Config import Config
from sklearn.pipeline import FeatureUnion
import itertools
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastsklearnfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import GroupByThenGenerator
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.DummyOneTransformation import DummyOneTransformation
import copy

class ExploreKitSelection_iterative_search:
    def __init__(self, dataset_config, classifier=LogisticRegression(), grid_search_parameters={'classifier__penalty': ['l2'],
                                                                                                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'classifier__solver': ['lbfgs'],
                                                                                                'classifier__class_weight': ['balanced'],
                                                                                                'classifier__max_iter': [10000]
                                                                                                }
                 ):
        self.dataset_config = dataset_config
        self.classifier = classifier
        self.grid_search_parameters = grid_search_parameters

    #generate all possible combinations of features
    def generate(self):

        s = Splitter(train_fraction=[0.6, 10000000], seed=42)
        #s = Splitter(train_fraction=[0.1, 10000000], seed=42)

        self.dataset = Reader(self.dataset_config[0], self.dataset_config[1], s)
        self.raw_features = self.dataset.read()

        #g = Generator(raw_features)
        #self.candidates = g.generate_all_candidates()
        #print("Number candidates: " + str(len(self.candidates)))

    #rank and select features
    def random_select(self, k: int):
        arr = np.arange(len(self.candidates))
        np.random.shuffle(arr)
        return arr[0:k]

    def generate_target(self):
        current_target = self.dataset.splitted_target['train']
        self.current_target = LabelEncoder().fit_transform(current_target)

    def evaluate(self, candidate, score=make_scorer(roc_auc_score, average='micro'), folds=10):
        parameters = self.grid_search_parameters


        if not isinstance(candidate, CandidateFeature):
            pipeline = Pipeline([('features',FeatureUnion(

                        [(p.get_name(), p.pipeline) for p in candidate]
                    )),
                ('classifier', self.classifier)
            ])
        else:
            pipeline = Pipeline([('features', FeatureUnion(
                [
                    (candidate.get_name(), candidate.pipeline)
                ])),
                 ('classifier', self.classifier)
                 ])

        result = {}

        clf = GridSearchCV(pipeline, parameters, cv=self.preprocessed_folds, scoring=score, iid=False, error_score='raise')
        clf.fit(self.dataset.splitted_values['train'], self.current_target)
        result['score'] = clf.best_score_
        result['hyperparameters'] = clf.best_params_

        return result




    def create_starting_features(self):
        Fi: List[RawFeature]= self.dataset.raw_features

        #materialize and numpyfy the features
        starting_feature_matrix = np.zeros((Fi[0].materialize()['train'].shape[0], len(Fi)))
        for f_index in range(len(Fi)):
            starting_feature_matrix[:, f_index] = Fi[f_index].materialize()['train']
        return starting_feature_matrix


    def evaluate_candidates(self, candidates):
        self.preprocessed_folds = []
        for train, test in StratifiedKFold(n_splits=10, random_state=42).split(self.dataset.splitted_values['train'], self.current_target):
            self.preprocessed_folds.append((train, test))

        pool = mp.Pool(processes=int(Config.get("parallelism")))
        results = pool.map(self.evaluate_single_candidate, candidates)
        return results

    '''
    def evaluate_candidates(self, candidates):
        self.preprocessed_folds = []
        for train, test in StratifiedKFold(n_splits=10, random_state=42).split(self.dataset.splitted_values['train'],
                                                                               self.current_target):
            self.preprocessed_folds.append((train, test))

        results = []
        for c in candidates:
            results.append(self.evaluate_single_candidate(c))
        return results

    '''


    def evaluate_single_candidate(self, candidate):
        result = {}
        time_start_gs = time.time()
        try:
            result = self.evaluate(candidate)
            #print("feature: " + str(candidate) + " -> " + str(new_score))
        except Exception as e:
            print(str(candidate) + " -> " + str(e))
            result['score'] = -1.0
            result['hyperparameters'] = {}
            pass
        result['candidate'] = candidate
        result['time'] = time.time() - time_start_gs
        return result


    '''
    def evaluate_single_candidate(self, candidate):
        new_score = -1.0
        new_score = self.evaluate(candidate)
        return new_score
    '''

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


    def generate_features(self, transformations: List[Transformation], features: List[CandidateFeature]) -> List[CandidateFeature]:
        generated_features: List[CandidateFeature] = []
        for t_i in transformations:
            for f_i in t_i.get_combinations(features):
                if t_i.is_applicable(f_i):
                    generated_features.append(CandidateFeature(copy.deepcopy(t_i), f_i)) # do we need a deep copy here?
                    #if output is multidimensional adapt here
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




    def get_features_from_identity_candidate(self, identity: CandidateFeature, my_list = set()):
        for p in identity.parents:
            if isinstance(p, RawFeature) or not isinstance(p.transformation, IdentityTransformation):
                my_list.add(str(p))
            else:
                self.get_features_from_identity_candidate(p, my_list)
        return my_list


    def generate_merge_for_combination(self, a: List[CandidateFeature], b: List[CandidateFeature]) -> Set[Set[CandidateFeature]]:
        # feature concat, but does not work
        #if not order_matters and not repetition_allowed:
        #    return [[x, y] for x, y in itertools.product(*[a, b]) if x != y]
        result_list: Set[Set[CandidateFeature]] = set()

        for a_i in range(len(a)):
            for b_i in range(len(b)):
                #if both had as last transformation Identity -> then tricky
                if not isinstance(a[a_i], RawFeature) and isinstance(a[a_i].transformation, IdentityTransformation) and \
                        not isinstance(b[b_i], RawFeature) and isinstance(b[b_i].transformation, IdentityTransformation):
                    #we have to check whether they intersect or not
                    #so we climb down the transformation pipeline and gather all concatenated features
                    if len(self.get_features_from_identity_candidate(a[a_i]).intersection(self.get_features_from_identity_candidate(b[b_i]))) == 0:
                        result_list.add(frozenset([a[a_i], b[b_i]]))
                else:
                    if str(a[a_i]) != str(b[b_i]):
                        result_list.add(frozenset([a[a_i], b[b_i]]))
        return result_list




    def run(self):
        # generate all candidates
        self.generate()
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        unary_transformations: List[UnaryTransformation] = []
        unary_transformations.append(PandasDiscretizerTransformation(number_bins=10))
        unary_transformations.append(MinMaxScalingTransformation())

        binary_transformations: List[Transformation] = []
        binary_transformations.extend(HigherOrderCommutativeClassGenerator(2, methods=[np.nansum, np.nanprod]).produce())
        binary_transformations.extend(NumpyBinaryClassGenerator(methods=[np.divide, np.subtract]).produce())

        binary_transformations.extend(GroupByThenGenerator(2, methods=[np.nanmax,
                                                                             np.nanmin,
                                                                             np.nanmean,
                                                                             np.nanstd]).produce())

        cost_2_raw_features: Dict[int, List[CandidateFeature]] = {}
        cost_2_unary_transformed: Dict[int, List[CandidateFeature]] = {}
        cost_2_binary_transformed: Dict[int, List[CandidateFeature]] = {}
        cost_2_combination: Dict[int, List[CandidateFeature]] = {}

        cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = {}

        complexity_delta = 1.0

        epsilon = 0.00
        limit_runs = 5  # 5

        baseline_score = self.evaluate_candidates([CandidateFeature(DummyOneTransformation(None), [self.raw_features[0]])])[0]['score']
        print("baseline: " + str(baseline_score))


        max_feature = CandidateFeature(IdentityTransformation(None), [self.raw_features[0]])
        max_feature.score = -2

        for c in range(1, limit_runs):
            current_layer: List[CandidateFeature] = []

            #0th
            if c == 1:
                current_layer.extend(self.raw_features)

            # first unary
            # we apply all unary transformation to all c-1 in the repo (except combinations and other unary?)
            unary_candidates_to_be_applied: List[CandidateFeature] = []
            if (c - 1) in cost_2_raw_features:
                unary_candidates_to_be_applied.extend(cost_2_raw_features[c - 1])
            if (c - 1) in cost_2_binary_transformed:
                unary_candidates_to_be_applied.extend(cost_2_binary_transformed[c - 1])

            current_layer.extend(self.generate_features(unary_transformations, unary_candidates_to_be_applied))

            #second binary
            #get length 2 partitions for current cost
            partition = self.get_length_2_partition(c)
            print(partition)

            #apply cross product from partitions
            binary_candidates_to_be_applied: List[CandidateFeature] = []
            for p in partition:
                lists_for_each_element: List[List[CandidateFeature]] = [[], []]
                for element in range(2):
                    if p[element] in cost_2_raw_features:
                        lists_for_each_element[element].extend(cost_2_raw_features[p[element]])
                    if p[element] in cost_2_unary_transformed:
                        lists_for_each_element[element].extend(cost_2_unary_transformed[p[element]])
                    if p[element] in cost_2_binary_transformed:
                        lists_for_each_element[element].extend(cost_2_binary_transformed[p[element]])

                for bt in binary_transformations:
                    list_of_combinations = self.generate_merge(lists_for_each_element[0], lists_for_each_element[1], bt.parent_feature_order_matters, bt.parent_feature_repetition_is_allowed)
                    for combo in list_of_combinations:
                        binary_candidates_to_be_applied.append(CandidateFeature(copy.deepcopy(bt), combo))
            current_layer.extend(binary_candidates_to_be_applied)

            #third: feature combinations
            #first variant: treat combination as a transformation
            #therefore, we can use the same partition as for binary data

            combinations_to_be_applied: List[CandidateFeature] = []
            for p in partition:
                lists_for_each_element: List[List[CandidateFeature]] = [[], []]
                for element in range(2):
                    if p[element] in cost_2_raw_features:
                        lists_for_each_element[element].extend(cost_2_raw_features[p[element]])
                    if p[element] in cost_2_unary_transformed:
                        lists_for_each_element[element].extend(cost_2_unary_transformed[p[element]])
                    if p[element] in cost_2_binary_transformed:
                        lists_for_each_element[element].extend(cost_2_binary_transformed[p[element]])
                    if p[element] in cost_2_combination:
                        lists_for_each_element[element].extend(cost_2_combination[p[element]])


                list_of_combinations = self.generate_merge_for_combination(lists_for_each_element[0], lists_for_each_element[1])
                for combo in list_of_combinations:
                    combinations_to_be_applied.append(CandidateFeature(IdentityTransformation(None), list(combo)))
            current_layer.extend(combinations_to_be_applied)

            print(len(current_layer))
            print(current_layer)

            #now evaluate all from this layer
            print("----------- Evaluation of " + str(len(current_layer)) + " features -----------")
            results = self.evaluate_candidates(current_layer)
            print("----------- Evaluation Finished -----------")

            #calculate whether we drop the evaluated candidate
            for result in results:
                candidate: CandidateFeature = result['candidate']
                candidate.score = result['score']

                #print(str(candidate) + " -> " + str(candidate.score))

                if candidate.score > max_feature.score:
                    max_feature = candidate

                #calculate original score
                original_score = baseline_score #or zero??
                if not isinstance(candidate, RawFeature):
                    original_score = max([p.score for p in candidate.parents])

                accuracy_delta = result['score'] - original_score

                if accuracy_delta / complexity_delta > epsilon:
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
                    if not c in cost_2_dropped_evaluated_candidates:
                        cost_2_dropped_evaluated_candidates[c]: List[CandidateFeature] = []
                    cost_2_dropped_evaluated_candidates[c].append(candidate)

            if c in cost_2_dropped_evaluated_candidates:
                print("From " + str(len(current_layer)) + " candidates, we dropped " + str(len(cost_2_dropped_evaluated_candidates[c])))
            else:
                print("From " + str(len(current_layer)) + " candidates, we dropped 0")


            print(max_feature)

            pickle.dump(cost_2_raw_features, open(Config.get("tmp.folder") + "/data_raw.p", "wb"))
            pickle.dump(cost_2_unary_transformed, open(Config.get("tmp.folder") + "/data_unary.p", "wb"))
            pickle.dump(cost_2_binary_transformed, open(Config.get("tmp.folder") + "/data_binary.p", "wb"))
            pickle.dump(cost_2_combination, open(Config.get("tmp.folder") + "/data_combination.p", "wb"))
            pickle.dump(cost_2_dropped_evaluated_candidates, open(Config.get("tmp.folder") + "/data_dropped.p", "wb"))





#statlog_heart.csv=/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv
#statlog_heart.target=13

if __name__ == '__main__':
    dataset = (Config.get('statlog_heart.csv'), int(Config.get('statlog_heart.target')))
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6)

    selector = ExploreKitSelection_iterative_search(dataset)
    #selector = ExploreKitSelection(dataset, KNeighborsClassifier(), {'n_neighbors': np.arange(3,10), 'weights': ['uniform','distance'], 'metric': ['minkowski','euclidean','manhattan']})

    selector.run()







