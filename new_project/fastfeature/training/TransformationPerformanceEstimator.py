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
from sklearn.preprocessing import LabelEncoder

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
        for m in method_list:
            feature_names.append(name + '_' + str(number_parents) + '_' + m.__name__)
        return feature_names


    def build_feature_matrix(self, transformation: UnaryTransformation, classifier=None, run_regression=False):
        # find applicable attributes
        applicable_attributes: List[CandidateFeature] = []
        attribute_dataset_pointers: List[int] = []
        for d_i in range(len(self.datasets)):
            for attribute in self.datasets[d_i].raw_features:
                if transformation.is_applicable([attribute]):
                    applicable_attributes.append(attribute)
                    attribute_dataset_pointers.append(d_i)
        print(len(applicable_attributes))

        # create metafeatures for attributes
        metafeatures = [len, stats.skew, np.nanmax, np.nanmin, np.nanmean, np.nanstd, np.nanvar]
        attribute_metafeature_names = self.create_feature_names('attribute', metafeatures)
        attribute_feature_matrix = np.zeros((len(applicable_attributes), len(metafeatures)))
        for attribute_i in range(len(applicable_attributes)):
            for feature_i in range(len(metafeatures)):
                attribute_feature_matrix[attribute_i, feature_i] = metafeatures[feature_i](applicable_attributes[attribute_i].materialize()['train']) # get statistic from transformed training data

        #calculate class distribution for each dataset
        list_of_target_distributions = []
        for d_i in range(len(self.datasets)):
            list_of_target_distributions.append(self.calculate_class_distributions(self.datasets[d_i].splitted_target['train']))

        # create metafeatures for labels
        target_metafeatures = [len, stats.skew, np.max, np.min, np.mean, np.std, np.var]
        target_metafeature_names = self.create_feature_names('target', target_metafeatures)
        label_feature_matrix = np.zeros((len(attribute_dataset_pointers), len(target_metafeatures)))
        for attribute_i in attribute_dataset_pointers:
            for feature_i in range(len(target_metafeatures)):
                label_feature_matrix[attribute_i, feature_i] = target_metafeatures[feature_i](list_of_target_distributions[attribute_dataset_pointers[attribute_i]])

        # todos
        # maybe add info gain
        # generate correlation between label and attribute

        score_list: List[float] = []

        working_indices = []
        #run classifier on feature and on transformation(feature)
        for attribute_i in range(len(applicable_attributes)):
            try:
                #run plain attribute
                cv_results = cross_validate(classifier,
                                            applicable_attributes[attribute_i].materialize()['train'].reshape(-1, 1),
                                            self.datasets[attribute_dataset_pointers[attribute_i]].splitted_target['train'],
                                            cv=10,
                                            scoring=make_scorer(roc_auc_score, average='micro'),
                                            return_train_score=False)
                raw_score = np.average(cv_results['test_score'])

                # run transformed attribute
                # think about hyperparameter tuning
                cv_results = cross_validate(classifier,
                                            CandidateFeature(transformation, [applicable_attributes[attribute_i]]).materialize()['train'].reshape(-1, 1),
                                            self.datasets[attribute_dataset_pointers[attribute_i]].splitted_target['train'],
                                            cv=10,
                                            scoring=make_scorer(roc_auc_score, average='micro'),
                                            return_train_score=False)
                transformed_score = np.average(cv_results['test_score'])

                score_list.append(transformed_score - raw_score)
                working_indices.append(attribute_i)
            except:
                pass

        print(score_list)

        # now, we can train a classifier to estimate whether a transformation is benefitial
        # we can return the most important features per transformation to understand what work and what does not work

        feature_matrix = np.hstack((attribute_feature_matrix, label_feature_matrix))[working_indices, :]

        print("training size: " + str(feature_matrix.shape))

        feature_names = []
        feature_names.extend(attribute_metafeature_names)
        feature_names.extend(target_metafeature_names)

        if run_regression:
            model = XGBoostRegressor()
            model.hyperparameter_optimization(feature_matrix, score_list, feature_names)
        else:
            model = XGBoostClassifier()
            model.hyperparameter_optimization(feature_matrix, np.array(score_list) > 0.0, feature_names)





    def build_feature_matrix_binary_transformation(self, transformation: Transformation, classifier=None, run_regression=False):
        #first generate possible inputs for binary transformations
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

            CandidateIdToDataset.extend([d_i]*len(Foi))

        # create metafeatures for attributes
        number_parents = transformation.number_parent_features
        metafeatures = [len, stats.skew, np.nanmax, np.nanmin, np.nanmean, np.nanstd, np.nanvar]
        #Todo: Create difference, correlation features
        attribute_metafeature_names = self.create_feature_names('attribute', metafeatures, number_parents)
        attribute_feature_list = []
        working_candidate_to_dataset_id = []
        working_candidates_ids = []
        for attribute_i in range(len(Foi_all)):
            attribute_feature_vector = np.zeros(len(metafeatures) * number_parents)
            for parent_i in range(number_parents):
                for feature_i in range(len(metafeatures)):
                    try:
                        attribute_feature_vector[feature_i + parent_i*len(metafeatures)] = metafeatures[feature_i](Foi_all[attribute_i].parents[parent_i].materialize()[
                                'train'])  # get statistic from transformed training data
                        attribute_feature_list.append(attribute_feature_vector)
                        working_candidate_to_dataset_id.append(CandidateIdToDataset[attribute_i])
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

        print("training size: " + str(feature_matrix.shape))


        # todos
        # maybe add info gain
        # generate correlation between label and attribute
        '''
        score_list: List[float] = []

        working_indices = []
        # run classifier on feature and on transformation(feature)
        for attribute_i in range(len(attribute_feature_matrix)):
            try:
                # run plain attribute
                # first hstack parent features
                parent_feature_list = []
                for p in Foi_all[candidate_feature_id[attribute_i]].parents:
                    parent_feature_list.append(p.materialize()['train'].reshape(-1, 1))

                current_plain_matrix = np.hstack(parent_feature_list)

                current_target = self.datasets[attribute_dataset_pointers[attribute_i]].splitted_target['train']
                current_target = LabelEncoder().fit_transform(current_target)

                
                cv_results = cross_validate(classifier,
                                            current_plain_matrix,
                                            current_target,
                                            cv=10,
                                            scoring=make_scorer(roc_auc_score, average='micro'),
                                            return_train_score=False)
                raw_score = np.average(cv_results['test_score'])
                

                materialized_feature = Foi_all[candidate_feature_id[attribute_i]].materialize()['train'].reshape(-1, 1)

                # run transformed attribute
                # think about hyperparameter tuning
                
                cv_results = cross_validate(classifier,
                                            materialized_feature,
                                            current_target,
                                            cv=10,
                                            scoring=make_scorer(roc_auc_score, average='micro'),
                                            return_train_score=False)
                transformed_score = np.average(cv_results['test_score'])
    
                score_list.append(transformed_score - raw_score)
                
                working_indices.append(attribute_i)

            except Exception as e:
                print(e)

        print(score_list)
        print(len(score_list))
        print(len(working_indices))
        '''

        '''

        # now, we can train a classifier to estimate whether a transformation is benefitial
        # we can return the most important features per transformation to understand what work and what does not work

        feature_matrix = feature_matrix[working_indices, :]

        print("training size: " + str(feature_matrix.shape))

        feature_names = []
        feature_names.extend(attribute_metafeature_names)
        feature_names.extend(target_metafeature_names)

        if run_regression:
            model = XGBoostRegressor()
            model.hyperparameter_optimization(feature_matrix, score_list, feature_names)
        else:
            model = XGBoostClassifier()
            model.hyperparameter_optimization(feature_matrix, np.array(score_list) > 0.0, feature_names)
        
        '''






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
    #est.build_feature_matrix(PandasDiscretizerTransformation(number_bins=10), LogisticRegression())
    #est.build_feature_matrix(MinMaxScalingTransformation(), LogisticRegression())

    est.build_feature_matrix_binary_transformation(HigherOrderCommutativeClassGenerator(2, methods=[np.nansum]).produce()[0], LogisticRegression())



