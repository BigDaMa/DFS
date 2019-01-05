from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.transformations.Transformation import Transformation
from typing import List
from fastfeature.transformations.UnaryTransformation import UnaryTransformation
from fastfeature.transformations.generators.NumpyClassGeneratorInvertible import NumpyClassGeneratorInvertible
from fastfeature.transformations.generators.NumpyClassGenerator import NumpyClassGenerator
from fastfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from fastfeature.transformations.generators.GroupByThenGenerator import GroupByThenGenerator
from fastfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
import numpy as np



class Generator:
    def __init__(self, raw_features: List[CandidateFeature]):
        self.Fi: List[CandidateFeature] = raw_features

    def generate_all_candidates(self):
        candidates = self.generate_candidates()
        candidates.extend(self.Fi)
        return candidates


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
                                                        np.nanstd,
                                                        len]).produce())

        print("unary transformations: " + str(len(unary_transformations)))
        print("higherorder transformations: " + str(len(higher_order_transformations)))


        print("Fi: " + str(len(self.Fi)))

        #for f_i in self.Fi:
        #    print(f_i.get_name())

        Fui = self.generate_features(unary_transformations, self.Fi)

        print("Fui: " + str(len(Fui)))

        #for f_i in Fui:
        #    print(f_i.get_name())


        Fi_and_Fui = []
        Fi_and_Fui.extend(self.Fi)
        Fi_and_Fui.extend(Fui)

        Foi = self.generate_features(higher_order_transformations, Fi_and_Fui)

        #for f_i in Foi:
        #    print(f_i.get_name())

        print("Foi: " + str(len(Foi)))

        Foui = self.generate_features(unary_transformations, Foi)

        print("Foui: " + str(len(Foui)))

        Fi_cand = []
        Fi_cand.extend(Fui)
        Fi_cand.extend(Foi)
        Fi_cand.extend(Foui)

        return Fi_cand

    def generate_features(self, transformations: List[Transformation], features: List[CandidateFeature]):
        generated_features: List[CandidateFeature] = []
        for t_i in transformations:
            for f_i in t_i.get_combinations(features):
                if t_i.is_applicable(f_i):
                    generated_features.append(CandidateFeature(t_i, f_i))
                    #if output is multidimensional adapt here
        return generated_features

    def materialize(self, features: List[CandidateFeature]):
        successful = 0
        for f in features:
            try:
                #print(f.get_name())
                f.materialize()
                successful += 1
            except Exception as e:
                print(e)

        print("successful transformations: " + str(successful))



if __name__ == '__main__':
    from fastfeature.reader.Reader import Reader
    from fastfeature.splitting.Splitter import Splitter
    import time

    s = Splitter(train_fraction=[0.6, 10000000])

    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)
    dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9)
    #dataset = ("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6)






    r = Reader(dataset[0], dataset[1], s)
    raw_features = r.read()


    g = Generator(raw_features)

    start_time = time.time()

    candidates = g.generate_candidates()

    candidate_generation = time.time()

    print("candidate generation time: " + str(candidate_generation - start_time))

    candidates.sort(reverse=False)
    print(candidates[0])
    print(candidates[-1])

    '''
    print(len(candidates))

    g.materialize(candidates)

    print("materialization time: " + str(time.time() - candidate_generation))
    '''

