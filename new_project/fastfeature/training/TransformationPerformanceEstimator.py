from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.transformations.Transformation import Transformation
from typing import List, Dict, Any
from fastfeature.transformations.UnaryTransformation import UnaryTransformation
from fastfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
import numpy as np
from fastfeature.reader.Reader import Reader
from fastfeature.splitting.Splitter import Splitter
from scipy import stats
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from fastfeature.training.model.XGBoostRegressor import XGBoostRegressor
from fastfeature.training.model.XGBoostClassifier import XGBoostClassifier
from fastfeature.candidate_generation.explorekit.Generator import Generator
from fastfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from sklearn.preprocessing import LabelEncoder
from fastfeature.training.active_learning.ActiveLearner import ActiveLearner

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


    def calculate_class_distributions(self, class_labels):
        class_count: Dict[Any, int] = {}
        for label_i in range(len(class_labels)):
            if not class_labels[label_i] in class_count:
                class_count[class_labels[label_i]] = 0
            class_count[class_labels[label_i]] += 1

        distribution = np.array(list(class_count.values()))
        distribution.sort()
        distribution = np.true_divide(distribution, np.sum(distribution))
        return distribution


    def create_feature_names(self, name, method_list, number_parents=1):
        feature_names: List[str] = []
        for n in range(number_parents):
            for m in method_list:
                feature_names.append(name + '_' + str(n) + '_' + m.__name__)
        return feature_names


    def create_candidates_for_binary_transformations(self, transformation: Transformation):
        # first generate possible inputs for binary transformations
        unary_transformations: List[UnaryTransformation] = []
        unary_transformations.append(PandasDiscretizerTransformation(number_bins=10))
        unary_transformations.append(MinMaxScalingTransformation())

        g = Generator(None)

        CandidateIdToDataset = []
        Foi_all: List[CandidateFeature] = []
        for d_i in range(len(self.datasets)):
            Fi = self.datasets[d_i].raw_features
            Fui = g.generate_features(unary_transformations, Fi)

            Fi_and_Fui = []
            Fi_and_Fui.extend(Fi)
            Fi_and_Fui.extend(Fui)

            Foi: List[CandidateFeature] = g.generate_features([transformation], Fi_and_Fui)
            Foi_all.extend(Foi)

            CandidateIdToDataset.extend([d_i] * len(Foi))

        return Foi_all, CandidateIdToDataset

    def create_candidates_for_unary_transformations(self, transformation: Transformation):
        # first generate possible inputs for unary transformations

        g = Generator(None)

        CandidateIdToDataset = []
        featureCandidates: List[CandidateFeature] = []
        for d_i in range(len(self.datasets)):
            Fi = self.datasets[d_i].raw_features
            Fui = g.generate_features([transformation], Fi)

            featureCandidates.extend(Fui)
            CandidateIdToDataset.extend([d_i] * len(Fui))

        return featureCandidates, CandidateIdToDataset


    def get_metadata_feature_matrix(self, transformation: Transformation, featureCandidates: List[CandidateFeature], candidateIdToDatasetId: List[int], classifier=None, run_regression=False):
        # create metafeatures for attributes
        number_parents = transformation.number_parent_features
        metafeatures = [len, stats.skew, np.nanmax, np.nanmin, np.nanmean, np.nanstd, np.nanvar]
        #Todo: Create difference, correlation features
        attribute_metafeature_names = self.create_feature_names('attribute', metafeatures, number_parents)
        attribute_feature_list = []
        working_candidate_to_dataset_id = []
        working_candidates_ids = []
        for attribute_i in range(len(featureCandidates)):
            try:
                attribute_feature_vector = np.zeros(len(metafeatures) * number_parents)
                for parent_i in range(number_parents):
                    for feature_i in range(len(metafeatures)):
                        attribute_feature_vector[feature_i + parent_i*len(metafeatures)] = metafeatures[feature_i](featureCandidates[attribute_i].parents[parent_i].materialize()['train'])  # get statistic from transformed training data
                attribute_feature_list.append(attribute_feature_vector)
                working_candidate_to_dataset_id.append(candidateIdToDatasetId[attribute_i])
                working_candidates_ids.append(attribute_i)
            except:
                pass

        attribute_feature_matrix = np.vstack(attribute_feature_list)

        print(attribute_feature_matrix.shape)

        # calculate class distribution for each dataset
        list_of_target_distributions = []
        for d_i in range(len(self.datasets)):
            list_of_target_distributions.append(
                self.calculate_class_distributions(self.datasets[d_i].splitted_target['train']))

        # create metafeatures for labels
        target_metafeatures = [len, stats.skew, np.max, np.min, np.mean, np.std, np.var]
        target_metafeature_names = self.create_feature_names('target', target_metafeatures)
        label_feature_matrix = np.zeros((len(attribute_feature_matrix), len(target_metafeatures)))
        for attribute_i in range(len(attribute_feature_matrix)):
            for feature_i in range(len(target_metafeatures)):
                label_feature_matrix[attribute_i, feature_i] = target_metafeatures[feature_i](
                    list_of_target_distributions[working_candidate_to_dataset_id[attribute_i]])

        feature_matrix = np.hstack((attribute_feature_matrix, label_feature_matrix))

        feature_names = []
        feature_names.extend(attribute_metafeature_names)
        feature_names.extend(target_metafeature_names)

        if number_parents == 2:
            # add absolute differences
            diff_matrix = np.subtract(attribute_feature_matrix[:, 0:len(metafeatures)], attribute_feature_matrix[:, len(metafeatures):attribute_feature_matrix.shape[1]])

            if not transformation.parent_feature_order_matters:
                diff_matrix = np.abs(diff_matrix)

            feature_matrix = np.hstack((feature_matrix, diff_matrix))
            feature_names.extend(self.create_feature_names('diff', metafeatures))


            '''
            # add correlations
            correlation_matrix = np.zeros((len(attribute_feature_matrix), 1))
            for wi in range(len(working_candidates_ids)):
                try:
                    correlation_matrix[wi, 0] = np.corrcoef(
                        featureCandidates[working_candidates_ids[wi]].parents[0].materialize()['train'].reshape(-1, 1),
                        featureCandidates[working_candidates_ids[wi]].parents[1].materialize()['train'].reshape(-1, 1))[0, 1]
                except:
                    pass
            feature_matrix = np.hstack((feature_matrix, correlation_matrix))
            feature_names.append('correlation')
            '''

        return feature_matrix, working_candidates_ids, working_candidate_to_dataset_id, feature_names





    def run_active_learning(self, feature_matrix, working_candidates_ids, working_candidate_to_dataset_id, feature_names):
        print("training size: " + str(feature_matrix.shape))

        al = ActiveLearner(self.datasets,
                           featureCandidates,
                           working_candidates_ids,
                           working_candidate_to_dataset_id,
                           feature_matrix,
                           feature_names,
                           60)
        al.run()







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
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpkIxskf_bank_data.csv", 16))
    ###data_collection.append(("/home/felix/datasets/ExploreKit/csv/vehicleNorm.csv", 100))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6))

    est = TransformationPerformanceEstimator(data_collection)
    est.gather_data()


    #transformation = PandasDiscretizerTransformation(number_bins=10)
    #transformation = MinMaxScalingTransformation()
    #featureCandidates, candidateIdToDatasetId = est.create_candidates_for_unary_transformations(transformation)


    transformation = HigherOrderCommutativeClassGenerator(2, methods=[np.nanprod]).produce()[0]
    #transformation =  NumpyBinaryClassGenerator(methods=[np.subtract]).produce()[0]
    featureCandidates, candidateIdToDatasetId = est.create_candidates_for_binary_transformations(transformation)

    feature_matrix, working_candidates_ids, working_candidate_to_dataset_id, feature_names = \
        est.get_metadata_feature_matrix(transformation, featureCandidates, candidateIdToDatasetId, LogisticRegression())

    est.run_active_learning(feature_matrix, working_candidates_ids, working_candidate_to_dataset_id, feature_names)




