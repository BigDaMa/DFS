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
from fastfeature.reader.Reader import Reader
from fastfeature.splitting.Splitter import Splitter
import time
from fastfeature.candidate_generation.explorekit.Generator import Generator
from fastfeature.candidates.RawFeature import RawFeature
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from fastfeature.plotting.plotter import cool_plotting
import pickle
from sklearn.model_selection import GridSearchCV
from scipy import stats
from fastfeature.training.model.XGBoostClassifier import XGBoostClassifier
from fastfeature.training.TransformationPerformanceEstimator import TransformationPerformanceEstimator
import xgboost as xgb


class ExploreKitSelection:
    def __init__(self, dataset_config, classifier=LogisticRegression()):
        self.dataset_config = dataset_config
        self.classifier = classifier

    #generate all possible combinations of features
    def generate(self):

        s = Splitter(train_fraction=[0.6, 10000000], seed=42)

        self.dataset = Reader(self.dataset_config[0], self.dataset_config[1], s)
        raw_features = self.dataset.read()

        g = Generator(raw_features)
        self.candidates: List[CandidateFeature] = g.generate_candidates()
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

    #evaluate Error(fi + f_cand) vs Error(fi)



    def evaluate(self, feature_matrix, score=make_scorer(roc_auc_score, average='micro'), folds=5):
        cv_results = cross_validate(self.classifier,
                                    feature_matrix,
                                    self.current_target,
                                    cv=folds,
                                    scoring=score,
                                    return_train_score=False)
        score = np.average(cv_results['test_score'])
        return score


    '''
    def evaluate(self, feature_matrix, score=make_scorer(roc_auc_score, average='micro'), folds=20):
        parameters = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        clf = GridSearchCV(self.classifier, parameters, cv=folds, scoring=score)
        clf.fit(feature_matrix, self.current_target)
        score = clf.best_score_
        return score
    '''



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

    def create_feature_names(self, name, method_list, number_parents=1):
        feature_names: List[str] = []
        for n in range(number_parents):
            for m in method_list:
                feature_names.append(name + '_' + str(n) + '_' + m.__name__)
        return feature_names

    def prune_by_classification(self, transformation=HigherOrderCommutativeClassGenerator(2, methods=[np.nanprod]).produce()[0]):

        #load model
        #model: XGBoostClassifier = pickle.load(open('/home/felix/phd/fastfeature_logs/model/model_nanprod_35min.p', "rb"))
        model: XGBoostClassifier = pickle.load(
            open('/home/felix/phd/fastfeature_logs/model/model_nanprod_60min.p', "rb"))

        est = TransformationPerformanceEstimator([self.dataset_config])
        est.gather_data()

        #transformation = PandasDiscretizerTransformation(number_bins=10)
        #transformation = MinMaxScalingTransformation()
        #transformation = HigherOrderCommutativeClassGenerator(2, methods=[np.nanprod]).produce()[0]
        #transformation =  NumpyBinaryClassGenerator(methods=[np.subtract]).produce()[0]

        applicable_candidates = []
        which_candidate_ids_are_applicable = []
        for c_i in range(len(self.candidates)):
            if self.candidates[c_i].transformation.name == transformation.name:
                applicable_candidates.append(self.candidates[c_i])
                which_candidate_ids_are_applicable.append(c_i)

        candidateIdToDatasetId = []
        candidateIdToDatasetId.extend([0] * len(applicable_candidates))

        print("candidates: " + str(len(applicable_candidates)))

        feature_matrix, working_candidates_ids, working_candidate_to_dataset_id, feature_names = \
        est.get_metadata_feature_matrix(transformation, applicable_candidates, candidateIdToDatasetId,
                                        LogisticRegression())

        print("feature_matrix: " + str(feature_matrix.shape))



        xgdmat = xgb.DMatrix(feature_matrix, feature_names=feature_names)
        predictions = model.model.predict(xgdmat)
        labels = predictions > 0.25
        print("prediction: " + str(predictions))
        print("prediction: " + str(labels))

        skip_id_list: List[int] = []
        for p_i in range(len(predictions)):
            if labels[p_i] == False:
                skip_id_list.append(which_candidate_ids_are_applicable[working_candidates_ids[p_i]])

        print(skip_id_list)

        #for s_i in skip_id_list:
        #    print(self.candidates[s_i])

        all_names = []
        for c_i in range(len(self.candidates)):
            all_names.append(self.candidates[c_i].get_name())

        all_data = {}
        all_data['names'] = all_names
        all_data['ids_to_be_skipped'] = skip_id_list
        pickle.dump(all_data, open("/tmp/ids_to_be_skipped.p", "wb"))












    def run(self):
        # generate all candidates
        self.generate()
        starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        self.prune_by_classification()







if __name__ == '__main__':
    dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9)
    # dataset = ("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6)

    selector = ExploreKitSelection(dataset)

    selector.run()



