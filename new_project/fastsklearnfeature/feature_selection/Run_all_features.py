from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Set
import time
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
from fastsklearnfeature.configuration.Config import Config
import itertools
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
import copy
from fastsklearnfeature.candidate_generation.feature_space.explorekit_transformations import get_transformation_for_feature_space
from fastsklearnfeature.feature_selection.evaluation.EvaluationFramework import EvaluationFramework
from fastsklearnfeature.feature_selection.openml_wrapper.pipeline2openml import candidate2openml
from fastsklearnfeature.transformations.generators.OneHotGenerator import OneHotGenerator
from fastsklearnfeature.transformations.ImputationTransformation import ImputationTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
import numpy as np
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from typing import List, Dict, Set
import pickle
import warnings
warnings.filterwarnings("ignore")

class Run_RawFeatures(EvaluationFramework):
    def __init__(self, dataset_config, classifier=XGBClassifier, grid_search_parameters={
        'classifier__min_child_weight': [1, 5, 10],
        'classifier__gamma': [0.5, 1, 1.5, 2, 5],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.02],
        'classifier__n_estimators': [600],
        'classifier__objective': ['binary:logistic'],
        'classifier__silent': [True]
        },
                 transformation_producer=get_transformation_for_feature_space,
                 score=make_scorer(f1_score, average='micro'),
                 reader=None,
                 folds=10,
                 max_complexity=3
                 ):
        self.dataset_config = dataset_config
        self.classifier = classifier
        self.grid_search_parameters = grid_search_parameters
        self.transformation_producer = transformation_producer
        self.score = score
        self.reader = reader
        self.folds = folds
        self.max_complexity=max_complexity




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




    def load_data_all(self, path):
        cost_2_raw_features: Dict[int, List[RawFeature]] = pickle.load(open(path + "/data_raw.p", "rb"))
        cost_2_unary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_unary.p", "rb"))
        cost_2_binary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_binary.p", "rb"))
        cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(
            open(path + "/data_dropped.p", "rb"))

        # build tree from logged data

        # get last layer:
        all_layers = list(cost_2_raw_features.keys())
        all_layers.extend(list(cost_2_unary_transformed.keys()))
        all_layers.extend(list(cost_2_binary_transformed.keys()))
        all_layers.extend(list(cost_2_dropped_evaluated_candidates.keys()))

        last_layer = max(all_layers)
        all_layers = None

        # create string2candidate dictionary
        def extend_string2candidate(my_dict: Dict[int, List[CandidateFeature]],
                                    string2candidate: Dict[str, CandidateFeature], last_layer):
            for c in range(0, last_layer + 1):
                if c in my_dict:
                    for v in my_dict[c]:
                        string2candidate[str(v)] = v

        string2candidate: Dict[str, CandidateFeature] = {}
        extend_string2candidate(cost_2_raw_features, string2candidate, last_layer)
        extend_string2candidate(cost_2_unary_transformed, string2candidate, last_layer)
        extend_string2candidate(cost_2_binary_transformed, string2candidate, last_layer)
        extend_string2candidate(cost_2_dropped_evaluated_candidates, string2candidate, last_layer)

        return string2candidate

    def get_interesting_features(self, string2candidate, complexity):

        def prune_from_root(c: CandidateFeature):
            if isinstance(c, RawFeature):
                return True
            if not all([prune_from_root(p) for p in c.parents]):
                return False
            if c.runtime_properties['score'] > max([p.runtime_properties['score'] for p in c.parents]):
                return True
            else:
                return False

        candidates: List[CandidateFeature] = []
        for c in string2candidate.values():
            if not isinstance(c, RawFeature): #and prune_from_root(c):
                if c.get_complexity() == complexity:
                    candidates.append(c)

        return candidates



    def run(self):
        self.global_starting_time = time.time()

        # generate all candidates
        self.generate(42)
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        myfolds = copy.deepcopy(list(self.preprocessed_folds))


        level_scores: Dict[int, List[float]] = {}
        level_test_scores: Dict[int, List[float]] = {}

        #string2candidate = self.load_data_all('/home/felix/phd/fastfeatures/results/eucalyptus')
        #string2candidate = self.load_data_all('/home/felix/phd/fastfeatures/results/contraceptive')
        #string2candidate = self.load_data_all('/home/felix/phd/fastfeatures/results/diabetes')
        #string2candidate = self.load_data_all('/home/felix/phd/fastfeatures/results/credit')
        #string2candidate = self.load_data_all('/home/felix/phd/fastfeatures/results/heart_new_all')
        #string2candidate = self.load_data_all('/tmp')


        baseline_features: List[CandidateFeature] = []
        for r in self.raw_features:
            if r.is_numeric() and not r.properties['categorical']:
                if not r.properties['missing_values']:
                    baseline_features.append(r)
                else:
                    baseline_features.append(CandidateFeature(ImputationTransformation(), [r]))
            else:
                baseline_features.extend([CandidateFeature(t, [r]) for t in OneHotGenerator(self.train_X_all, [r]).produce()])


        #baseline_features.extend(self.get_interesting_features('/home/felix/phd/fastfeatures/results/heart_small', 24))
        #baseline_features.extend(self.get_interesting_features('/home/felix/phd/fastfeatures/results/heart_new_all', 10))
        #baseline_features.extend(self.get_interesting_features(string2candidate, 2))




        '''
        for c in baseline_features:
            if isinstance(c, RawFeature):
                print(str(c) + " complexity: " + str(c.get_complexity()))
            else:
                print('nr: ' + str(c) + " complexity: " + str(c.get_complexity()))+
        '''


        # standardize
        scaled_baseline_features = []
        for c in baseline_features:
            scaled_baseline_features.append(CandidateFeature(MinMaxScalingTransformation(), [c]))

        #scaled_baseline_features = baseline_features

        combo = CandidateFeature(IdentityTransformation(len(baseline_features)), scaled_baseline_features)


        results = self.evaluate_candidates_detail([combo], myfolds, int(Config.get_default("parallelism", mp.cpu_count())))

        print(str(results[0].runtime_properties))


#statlog_heart.csv=/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv
#statlog_heart.target=13

if __name__ == '__main__':
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)

    #dataset = (Config.get('data_path') + "/phpn1jVwe_mammography.csv", 6)
    # dataset = (Config.get('data_path') + "/dataset_23_cmc_contraceptive.csv", 9)
    #dataset = (Config.get('data_path') + "/dataset_31_credit-g_german_credit.csv", 20)
    #dataset = (Config.get('data_path') + '/dataset_53_heart-statlog_heart.csv', 13)
    #dataset = (Config.get('data_path') + '/ILPD.csv', 10)
    # dataset = (Config.get('data_path') + '/iris.data', 4)
    # dataset = (Config.get('data_path') + '/data_banknote_authentication.txt', 4)
    # dataset = (Config.get('data_path') + '/ecoli.data', 8)
    #dataset = (Config.get('data_path') + '/breast-cancer.data', 0)
    #dataset = (Config.get('data_path') + '/transfusion.data', 4)
    # dataset = (Config.get('data_path') + '/test_categorical.data', 4)
    # dataset = ('../configuration/resources/data/transfusion.data', 4)
    #dataset = (Config.get('data_path') + '/wine.data', 0)

    from fastsklearnfeature.reader.OnlineOpenMLReader import OnlineOpenMLReader
    from fastsklearnfeature.feature_selection.openml_wrapper.openMLdict import openMLname2task

    #task_id = openMLname2task['transfusion'] #interesting
    # task_id = openMLname2task['iris']
    #task_id = openMLname2task['ecoli']
    #task_id = openMLname2task['breast cancer']
    #task_id = openMLname2task['contraceptive']
    task_id = openMLname2task['german credit'] #interesting
    # task_id = openMLname2task['monks']
    #task_id = openMLname2task['banknote']
    #task_id = openMLname2task['heart-statlog']
    # task_id = openMLname2task['musk']
    #task_id = openMLname2task['eucalyptus']
    #task_id = openMLname2task['haberman']
    #task_id = openMLname2task['quake']
    #task_id = openMLname2task['volcanoes']
    #task_id = openMLname2task['analcatdata']
    #task_id = openMLname2task['credit approval']
    #task_id = openMLname2task['lupus']
    #task_id = openMLname2task['diabetes']

    #task_id = openMLname2task['covertype']
    # task_id = openMLname2task['eeg_eye_state']
    #task_id = openMLname2task['MagicTelescope']
    #task_id = openMLname2task['mushroom']

    dataset = None

    selector = Run_RawFeatures(dataset, reader=OnlineOpenMLReader(task_id, 1), score=make_scorer(roc_auc_score), max_complexity=3) #make_scorer(f1_score, average='micro') #make_scorer(roc_auc_score)
    #selector = Run_RawFeatures(dataset, score=make_scorer(roc_auc_score))
    #selector = ExploreKitSelection(dataset, KNeighborsClassifier(), {'n_neighbors': np.arange(3,10), 'weights': ['uniform','distance'], 'metric': ['minkowski','euclidean','manhattan']})

    selector.run()







