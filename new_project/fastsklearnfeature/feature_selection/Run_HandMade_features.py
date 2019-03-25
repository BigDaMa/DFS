from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from typing import List
import time
from sklearn.linear_model import LogisticRegression
import numpy as np
from fastsklearnfeature.configuration.Config import Config
import itertools

from fastsklearnfeature.transformations.GroupByThenTransformation import GroupByThenTransformation
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.binary.NonCommutativeBinaryTransformation import NonCommutativeBinaryTransformation
from fastsklearnfeature.transformations.HigherOrderCommutativeTransformation import HigherOrderCommutativeTransformation

from fastsklearnfeature.feature_selection.evaluation.EvaluationFramework import EvaluationFramework


class ExploreKitSelection_iterative_search(EvaluationFramework):
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

        all_representations = set()
        partitions = self.partition(x)

        #get candidates of partitions
        candidates_with_cost_x = {}
        for i in range(x+1):
            candidates_with_cost_x[i] = self.get_all_features_equal_n_cost(i)

        for p in partitions:
            current_list = itertools.product(*[candidates_with_cost_x[pi] for pi in p])
            for c_output in current_list:
                if len(set(c_output)) == len(p):
                    all_representations.add(frozenset(c_output))

        return all_representations


    def filter_failing_features(self):
        working_features: List[CandidateFeature] = []
        for candidate in self.candidates:
            try:
                candidate.fit(self.dataset.splitted_values['train'])
                candidate.transform(self.dataset.splitted_values['train'])
            except:
                continue
            working_features.append(candidate)
        return working_features



    def explorekit_heart_features(self, name2feature):
        explore_kit_features = []
        explore_kit_features.extend(self.raw_features)

        # Discretize({Mean(age) GROUP BY Discretize(sex), Discretize(exercise_induced_angina)})
        discr_sex = CandidateFeature(PandasDiscretizerTransformation(10), [name2feature['sex']])
        discr_angina = CandidateFeature(PandasDiscretizerTransformation(10), [name2feature['exercise_induced_angina']])
        grouped = CandidateFeature(GroupByThenTransformation(np.mean, 3),
                                   [name2feature['age'], discr_sex, discr_angina])
        final = CandidateFeature(PandasDiscretizerTransformation(10), [grouped])

        explore_kit_features.append(final)

        all_f = CandidateFeature(IdentityTransformation(len(explore_kit_features)), explore_kit_features)
        return [all_f]



    #Index(['Recency', 'Frequency', 'Monetary', 'Time', 'Frequency/Time',
    #   'Recency**2/Time', 'Frequency**2/Time', 'Time/Frequency'],
    #  dtype='object')
    def sisso_transfusion_features(self, name2feature):
        sisso_features = []
        sisso_features.extend(self.raw_features)

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide), [name2feature['Frequency'], name2feature['Time']]))

        squared_recency = CandidateFeature(HigherOrderCommutativeTransformation(np.prod, 2), [name2feature['Recency'], name2feature['Recency']])
        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [squared_recency, name2feature['Time']]))

        squared_frequency = CandidateFeature(HigherOrderCommutativeTransformation(np.prod, 2),
                                           [name2feature['Frequency'], name2feature['Frequency']])
        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [squared_frequency, name2feature['Time']]))

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [name2feature['Time'], name2feature['Frequency']]))

        all_f = CandidateFeature(IdentityTransformation(len(sisso_features)), sisso_features)
        return [all_f]

    # Index(['Recency', 'Frequency', 'Monetary', 'Time', 'Frequency/Time'],
    def sisso_transfusion_features_new(self, name2feature):
        sisso_features = []
        sisso_features.extend(self.raw_features)

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [name2feature['Frequency'], name2feature['Time']]))

        all_f = CandidateFeature(IdentityTransformation(len(sisso_features)), sisso_features)
        return [all_f]

    #Index(['Recency', 'Frequency', 'Monetary', 'Time', 'Time/Monetary','Frequency**2/Time', 'Monetary/Time']
    def sisso_transfusion_features_new2(self, name2feature):
        sisso_features = []
        sisso_features.extend(self.raw_features)

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide), [name2feature['Time'], name2feature['Monetary']]))

        squared_Frequency = CandidateFeature(HigherOrderCommutativeTransformation(np.prod, 2), [name2feature['Frequency'], name2feature['Frequency']])
        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [squared_Frequency, name2feature['Time']]))

        squared_frequency = CandidateFeature(HigherOrderCommutativeTransformation(np.prod, 2),
                                           [name2feature['Frequency'], name2feature['Frequency']])
        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [squared_frequency, name2feature['Time']]))

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [name2feature['Monetary'], name2feature['Time']]))

        all_f = CandidateFeature(IdentityTransformation(len(sisso_features)), sisso_features)
        return [all_f]

    # Index(['Recency', 'Frequency', 'Monetary', 'Time', 'Recency**2/Time',
    #        'Recency/Time', 'Monetary**2/Time', 'Monetary/Time'],
    def sisso_transfusion_features_new3(self, name2feature) -> List[CandidateFeature]:
        sisso_features = []
        sisso_features.extend(self.raw_features)

        squared_recency = CandidateFeature(HigherOrderCommutativeTransformation(np.prod, 2),
                                           [name2feature['Recency'], name2feature['Recency']])

        squared_monetary = CandidateFeature(HigherOrderCommutativeTransformation(np.prod, 2),
                                             [name2feature['Monetary'], name2feature['Monetary']])

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [name2feature['Recency'], name2feature['Time']]))

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [name2feature['Monetary'], name2feature['Time']]))

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [squared_monetary, name2feature['Time']]))

        sisso_features.append(CandidateFeature(NonCommutativeBinaryTransformation(np.divide),
                                               [squared_recency, name2feature['Time']]))

        all_f = CandidateFeature(IdentityTransformation(len(sisso_features)), sisso_features)
        return [all_f]

    def run(self):
        self.global_starting_time = time.time()

        # generate all candidates
        self.generate()
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()


        #working_features = self.filter_failing_features()
        #all_f = CandidateFeature(IdentityTransformation(len(working_features)), working_features)


        name2feature = {}
        for f in self.raw_features:
            name2feature[str(f)] = f

        #my_list = self.explorekit_heart_features(name2feature)

        my_list = self.sisso_transfusion_features_new(name2feature)

        results = self.evaluate_candidates(my_list)

        print(results)

        print("Complexity: " + str(my_list[0].get_complexity()))

        for r in range(len(results)):
            print("(" + str(r+1) + "," + str(results[r]['test_score']) + ")")




#statlog_heart.csv=/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv
#statlog_heart.target=13

if __name__ == '__main__':
    #dataset = (Config.get('statlog_heart.csv'), int(Config.get('statlog_heart.target')))
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6)

    dataset = (Config.get('transfusion.csv'), 4)

    selector = ExploreKitSelection_iterative_search(dataset)
    #selector = ExploreKitSelection(dataset, KNeighborsClassifier(), {'n_neighbors': np.arange(3,10), 'weights': ['uniform','distance'], 'metric': ['minkowski','euclidean','manhattan']})

    selector.run()





