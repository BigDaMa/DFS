from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.transformations.Transformation import Transformation
from typing import List, Dict
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastsklearnfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from fastsklearnfeature.transformations.generators.GroupByThenGenerator import GroupByThenGenerator
from fastsklearnfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.reader.Reader import Reader
import numpy as np
import copy
import itertools
from fastsklearnfeature.candidate_generation.tree.FeatureNode import FeatureNode
#import networkx as nx
from fastsklearnfeature.configuration.Config import Config
import multiprocessing as mp



class TreeGenerator:
    def __init__(self, raw_features: List[CandidateFeature]):
        self.Fi: List[CandidateFeature] = raw_features

    def generate_all_candidates(self):
        candidates = self.generate_candidates()
        candidates.extend(self.Fi)

        self.candidates = candidates

        return candidates


    def plot_graph(self, graph):
        temp_graph = copy.deepcopy(graph)

        for n in temp_graph.nodes:
            temp_graph.node[n]['feature'] = str("")

        nx.write_graphml(temp_graph, '/tmp/tree_generator_new.graphml')



    def generate_for_transformation(self, t_i):
        result_features = []
        for f_i in t_i.get_combinations(list(itertools.chain(*self.current_features))):
            if t_i.is_applicable(f_i):
                current_feature = CandidateFeature(copy.deepcopy(t_i), f_i)
                #print(current_feature)

                result_features.append(current_feature)
        return result_features


    def generate_in_parallel(self, transformations, current_features):
        self.current_features = current_features
        pool = mp.Pool(processes=int(Config.get("parallelism")))
        results = pool.map(self.generate_for_transformation, transformations)

        return list(itertools.chain(*results))



    def generate_candidates(self):


        unary_transformations: List[UnaryTransformation] = []
        unary_transformations.append(PandasDiscretizerTransformation(number_bins=10))
        unary_transformations.append(MinMaxScalingTransformation())


        higher_order_transformations: List[Transformation] = []
        higher_order_transformations.extend(HigherOrderCommutativeClassGenerator(2, methods=[np.nansum, np.nanprod]).produce())
        higher_order_transformations.extend(NumpyBinaryClassGenerator(methods=[np.divide, np.subtract]).produce())

        #count is missing
        higher_order_transformations.extend(GroupByThenGenerator(2, methods=[np.nanmax,
                                                        np.nanmin,
                                                        np.nanmean,
                                                        np.nanstd]).produce())

        transformations = []
        transformations.extend(unary_transformations)
        transformations.extend(higher_order_transformations)
        #transformations.append(IdentityTransformation(2))




        print("unary transformations: " + str(len(unary_transformations)))
        print("higherorder transformations: " + str(len(higher_order_transformations)))

        features = self.Fi


        '''
        graph = nx.DiGraph()


        graph.add_node('root')
        for f in features:
            graph.add_node(str(f))
            graph.node[str(f)]['feature'] = f
            graph.add_edge('root', str(f))
        '''

        F0 = features

        F = []
        F.append(F0)


        '''
        for depth in range(2):
            F_t_plus_1 = []
            for t_i in transformations:
                for f_i in t_i.get_combinations(list(itertools.chain(*F[0:depth+1]))):
                    if t_i.is_applicable(f_i):
                        current_feature = CandidateFeature(copy.deepcopy(t_i), f_i)
                        print(current_feature)

                        
                        graph.add_node(str(current_feature))
                        graph.node[str(current_feature)]['feature'] = current_feature
                        for parent_feature in f_i:
                            graph.add_edge(str(parent_feature), str(current_feature))
                        
                        F_t_plus_1.append(current_feature)
            F.append(F_t_plus_1)

            print(len(list(itertools.chain(*F))))

        #self.plot_graph(graph)
        '''

        for depth in range(3):
            results = self.generate_in_parallel(transformations, F[0:depth + 1])
            F.append(results)

            print(len(list(itertools.chain(*F))))

        # self.plot_graph(graph)








    def generate_features(self, transformations: List[Transformation], features: List[CandidateFeature]):
        generated_features: List[CandidateFeature] = []
        for t_i in transformations:
            for f_i in t_i.get_combinations(features):
                if t_i.is_applicable(f_i):
                    generated_features.append(CandidateFeature(copy.deepcopy(t_i), f_i))
                    #if output is multidimensional adapt here
        return generated_features


    def materialize(self, features: List[CandidateFeature], r: Reader):
        successful = 0
        for f in features:
            try:
                #print(f.get_name())
                f.fit(r.splitted_values['train'])
                f.transform(r.splitted_values['train'])
                f.transform(r.splitted_values['valid'])
                f.transform(r.splitted_values['test'])
                successful += 1
            except Exception as e:
                print(str(f) + " -> " + str(e))

        print("successful transformations: " + str(successful))


    def get_all_features_equal_n_cost(self, cost):
        filtered_candidates = []
        for i in range(len(self.candidates)):
            if (self.candidates[i].get_number_of_transformations() + 1) == cost:
                filtered_candidates.append(self.candidates[i])
        return filtered_candidates

    # https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    def partition(self, number):
        answer = set()
        answer.add((number,))
        for x in range(1, number):
            for y in self.partition(number - x):
                answer.add(tuple(sorted((x,) + y)))
        return answer


    def get_all_possible_representations_for_step_x(self, x):

        all_representations = set()
        partitions = self.partition(x)



        #get candidates of partitions
        candidates_with_cost_x = {}
        for i in range(x+1):
            candidates_with_cost_x[i] = self.get_all_features_equal_n_cost(i)


        print(partitions)


        for p in partitions:
            current_list = itertools.product(*[candidates_with_cost_x[pi] for pi in p])
            for c_output in current_list:
                if len(set(c_output)) == len(p):
                    all_representations.add(frozenset(c_output))




        print(len(all_representations))

        return all_representations


if __name__ == '__main__':
    from fastsklearnfeature.splitting.Splitter import Splitter
    import time

    s = Splitter(train_fraction=[0.6, 10000000])

    dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6)






    r = Reader(dataset[0], dataset[1], s)
    raw_features = r.read()


    g = TreeGenerator(raw_features)

    start_time = time.time()

    g.generate_candidates()




