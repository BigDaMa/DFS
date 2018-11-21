from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.transformations.Transformation import Transformation
from typing import List
from fastfeature.transformations.UnaryTransformation import UnaryTransformation
from fastfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
import numpy as np
from fastfeature.reader.Reader import Reader
from fastfeature.splitting.Splitter import Splitter
from scipy import stats



class TransformationPerformanceEstimator:
    def __init__(self, datasets_config, splitter: Splitter=Splitter()):
        self.datasets_config = datasets_config
        self.splitter = splitter

    def gather_data(self):
        self.datasets: List[Reader] = []
        for (file_name, target_column) in self.datasets_config:
            print(file_name)
            r = Reader(file_name, target_column, self.splitter)
            r.read()
            self.datasets.append(r)

    def build_feature_matrix(self, transformation: UnaryTransformation, features, classifier):
        applicable_attributes: List[CandidateFeature] = []
        for d in self.datasets:
            for attribute in d.raw_features:
                if transformation.is_applicable([attribute]):
                    applicable_attributes.append(attribute)
        print(len(applicable_attributes))

        #run statistics over attributes
        feature_matrix = np.zeros((len(applicable_attributes), len(features)))
        for attribute_i in range(len(applicable_attributes)):
            for feature_i in range(len(features)):
                feature_matrix[attribute_i, feature_i] = features[feature_i](applicable_attributes[attribute_i].materialize()['train'])


        #create features for labels

        #create labels / run classifier on feature and on feature + transformation





    def train(self):

        unary_transformations: List[UnaryTransformation] = []
        unary_transformations.append(PandasDiscretizerTransformation(number_bins=10))
        unary_transformations.append(MinMaxScalingTransformation())


if __name__ == '__main__':
    data_collection = []
    #data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9))
    #data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpkIxskf_bank_data.csv", 16))
    #data_collection.append(("/home/felix/datasets/ExploreKit/csv/vehicleNorm.csv", 100))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6))

    est = TransformationPerformanceEstimator(data_collection)
    est.gather_data()
    est.build_feature_matrix(PandasDiscretizerTransformation(number_bins=10),
                             [stats.skew, np.nanmax, np.nanmin, np.nanmean, np.nanstd, np.nanvar])



