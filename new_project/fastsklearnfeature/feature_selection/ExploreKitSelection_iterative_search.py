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
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from fastsklearnfeature.configuration.Config import Config
from sklearn.pipeline import FeatureUnion
import itertools

class ExploreKitSelection_iterative_search:
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

    def get_all_features_equal_n_cost(self, cost):
        filtered_candidates = []
        for i in range(len(self.candidates)):
            if (self.candidates[i].get_number_of_transformations() + 1) == cost:
                filtered_candidates.append(self.candidates[i])
        return filtered_candidates



    def get_all_possible_representations_for_step_x(self, x):

        all_representations = []
        partitions = self.partition(x)

        #get candidates of partitions
        candidates_with_cost_x = {}
        for i in range(x+1):
            candidates_with_cost_x[i] = self.get_all_features_equal_n_cost(i)

        for p in partitions:
            all_representations.extend(itertools.product(*[candidates_with_cost_x[pi] for pi in p]))

        return all_representations


    def run(self):
        # generate all candidates
        self.generate()
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()


        all_results = []


        found_features = []
        for round in range(4):
            start_time = time.time()


            all_current_rep = self.get_all_possible_representations_for_step_x(round + 1)
            print(len(all_current_rep))

            results = self.evaluate_candidates(all_current_rep)

            all_results.append(results)

            new_scores = [r['score'] for r in results]
            best_id = np.argmax(new_scores)

            found_features.append(results[best_id])

            print(found_features)

            print("evaluation time: " + str((time.time()-start_time) / 60) + " min")



        return all_results



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

    results = selector.run()

    pickle.dump(results, open("/tmp/all_data_iterations.p", "wb"))





