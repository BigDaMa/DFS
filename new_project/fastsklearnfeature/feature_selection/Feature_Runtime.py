from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.transformations.Transformation import Transformation
from typing import List
import numpy as np
from fastsklearnfeature.reader.Reader import Reader
from fastsklearnfeature.splitting.Splitter import Splitter
import time
from fastsklearnfeature.candidate_generation.explorekit.Generator import Generator
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from fastfeature.plotting.plotter import cool_plotting
import pickle
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from fastsklearnfeature.configuration.Config import Config

class ExploreKitSelection:
    def __init__(self, dataset_config, classifier=LogisticRegression(), grid_search_parameters={'classifier__penalty': ['l2'],
                                                                                                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'classifier__solver': ['lbfgs']}):
        self.dataset_config = dataset_config
        self.classifier = classifier
        self.grid_search_parameters = grid_search_parameters

    #generate all possible combinations of features
    def generate(self):

        s = Splitter(train_fraction=[0.6, 10000000], seed=42)
        #s = Splitter(train_fraction=[0.1, 10000000], seed=42)

        self.dataset = Reader(self.dataset_config[0], self.dataset_config[1], s)
        raw_features = self.dataset.read()

        g = Generator(raw_features)
        self.candidates = g.generate_all_candidates()
        print("Number candidates: " + str(len(self.candidates)))

    #rank and select features
    def random_select(self, k: int):
        arr = np.arange(len(self.candidates))
        np.random.shuffle(arr)
        return arr[0:k]

    def select_interpretable(self, k: int):
        inv_map = {v: k for k, v in self.candidate_id_to_ranked_id.items()}
        selected = []
        for i in range(k):
            selected.append(inv_map[i])
        return selected


    def generate_target(self):
        current_target = self.dataset.splitted_target['train']
        self.current_target = LabelEncoder().fit_transform(current_target)

    def evaluate(self, candidate: CandidateFeature, runs=10):
        for i in range(runs):
            candidate.pipeline.fit(self.dataset.splitted_values['train'])
            candidate.pipeline.transform(self.dataset.splitted_values['train'])




    def create_starting_features(self):
        Fi: List[RawFeature]= self.dataset.raw_features

        #materialize and numpyfy the features
        starting_feature_matrix = np.zeros((Fi[0].materialize()['train'].shape[0], len(Fi)))
        for f_index in range(len(Fi)):
            starting_feature_matrix[:, f_index] = Fi[f_index].materialize()['train']
        return starting_feature_matrix

    def my_arg_sort(self, seq):
        # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
        # non-lambda version by Tony Veijalainen
        return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

    def get_interpretability_ranking(self):
        #high interpretability -> low interpretability
        interpretability_ids = self.my_arg_sort(self.candidates)

        self.candidate_id_to_ranked_id = {}
        for i in range(len(interpretability_ids)):
            self.candidate_id_to_ranked_id[interpretability_ids[i]] = i

    def get_traceability_ranking(self):
        # high traceability -> low traceability
        self.traceability: List[float] = []
        for c_i in range(len(self.candidates)):
            self.traceability.append(self.candidates[c_i].calculate_traceability())
        ids = np.argsort(np.array(self.traceability)*-1)

        self.candidate_id_to_ranked_id = {}
        for i in range(len(ids)):
            self.candidate_id_to_ranked_id[ids[i]] = i

        all_data = {}
        all_data['my_dict'] = self.candidate_id_to_ranked_id
        all_data['traceability'] = self.traceability
        pickle.dump(all_data, open("/tmp/traceability.p", "wb"))


    def get_interpretability(self, candidate_id):
        return 1.0 - ((self.candidate_id_to_ranked_id[candidate_id] + 1) / float(len(self.candidate_id_to_ranked_id)))


    def evaluate_candidates(self, candidates):
        self.preprocessed_folds = []
        for train, test in StratifiedKFold(n_splits=10, random_state=42).split(self.dataset.splitted_values['train'], self.current_target):
            self.preprocessed_folds.append((train, test))

        pool = mp.Pool(processes=1)
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
        runs = 10
        try:
            self.evaluate(candidate, runs)
            #print("feature: " + str(candidate) + " -> " + str(new_score))
        except Exception as e:
            print(str(candidate) + " -> " + str(e))
            pass
        result['candidate'] = candidate
        result['time'] = (time.time() - time_start_gs) / float(runs)
        return result


    '''
    def evaluate_single_candidate(self, candidate):
        new_score = -1.0
        new_score = self.evaluate(candidate)
        return new_score
    '''


    def run(self):
        # generate all candidates
        self.generate()
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        candidate_name_to_id = {}
        for c_i in range(len(self.candidates)):
            candidate_name_to_id[self.candidates[c_i].get_name()] = c_i

        pickle.dump(candidate_name_to_id, open("/tmp/name2id.p", "wb"))

        pickle.dump(self.candidates, open("/tmp/all_candiates.p", "wb"))

        self.get_interpretability_ranking()
        #self.get_traceability_ranking()

        #evaluate starting matrix
        #start_score = self.evaluate(starting_feature_matrix)
        start_score = -1
        print("start score: " + str(start_score))

        #get candidates that should be evaluated
        ranked_selected_candidate_ids = self.select_interpretable(len(self.candidates))

        start_time = time.time()

        results = self.evaluate_candidates(np.array(self.candidates)[ranked_selected_candidate_ids])

        print("evaluation time: " + str((time.time()-start_time) / 60) + " min")

        return start_score, results, ranked_selected_candidate_ids



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

    selector = ExploreKitSelection(dataset)
    #selector = ExploreKitSelection(dataset, KNeighborsClassifier(), {'n_neighbors': np.arange(3,10), 'weights': ['uniform','distance'], 'metric': ['minkowski','euclidean','manhattan']})

    start_score, results, ids = selector.run()

    pickle.dump(results, open("/tmp/all_data_runtime.p", "wb"))





