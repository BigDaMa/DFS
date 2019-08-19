from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Set
import time
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
from fastsklearnfeature.configuration.Config import Config
import itertools
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.ImputationTransformation import ImputationTransformation
from fastsklearnfeature.transformations.generators.OneHotGenerator import OneHotGenerator
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from fastsklearnfeature.transformations.StandardScalingTransformation import StandardScalingTransformation
import copy
from fastsklearnfeature.feature_selection.evaluation.EvaluationFramework import EvaluationFramework
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.feature_selection.openml_wrapper.pipeline2openml import candidate2openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class Run_RawFeatures(EvaluationFramework):
    def __init__(self, dataset_config, classifier=LogisticRegression, grid_search_parameters={'classifier__penalty': ['l2'],
                                                                                                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'classifier__solver': ['lbfgs'],
                                                                                                'classifier__class_weight': ['balanced'],
                                                                                                'classifier__max_iter': [10000],
                                                                                                'classifier__multi_class':['auto']
                                                                                                },
                 score=make_scorer(f1_score, average='micro'),
                 reader=None,
                 folds=10
                 ):
        self.dataset_config = dataset_config
        self.classifier = classifier
        self.grid_search_parameters = grid_search_parameters
        self.score = score
        self.reader = reader
        self.folds = folds

    def run(self):
        self.global_starting_time = time.time()

        # generate all candidates
        self.generate(42)
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        myfolds = copy.deepcopy(list(self.preprocessed_folds))


        baseline_features: List[CandidateFeature] = []
        for r in self.raw_features:
            if r.is_numeric() and (not 'categorical' in r.properties or not r.properties['categorical']):
                if not r.properties['missing_values']:
                    baseline_features.append(r)
                else:
                    baseline_features.append(CandidateFeature(ImputationTransformation(), [r]))
            else:
                baseline_features.extend([CandidateFeature(t, [r]) for t in OneHotGenerator(self.train_X_all, [r]).produce()])

        #scale everything
        for bf_i in range(len(baseline_features)):
            baseline_features[bf_i] = CandidateFeature(StandardScalingTransformation(), [baseline_features[bf_i]])


        print(len(baseline_features))

        combo = CandidateFeature(IdentityTransformation(len(baseline_features)), baseline_features)
        '''
        categorical_ids = []
        for r in self.raw_features:
            if 'categorical' in r.properties and r.properties['categorical']:
                categorical_ids.append(r.column_id)

        combo = CandidateFeature(IdentityTransformation(0), self.raw_features)
        if len(categorical_ids) >= 1:
            combo.pipeline = Pipeline(steps=[('imputation', SimpleImputer(strategy='mean')),
                                         ('onehot', OneHotEncoder(categorical_features=categorical_ids)), ('scaling', StandardScaler(with_mean=False))])
        else:
            combo.pipeline = Pipeline(steps=[('imputation', SimpleImputer(strategy='mean')), ('scaling', StandardScaler(with_mean=False))])
        '''

        results = self.evaluate_candidates([combo], myfolds)

        #print(results[0].runtime_properties)

        #candidate2openml(results[0], self.classifier, self.reader.task, 'RawFeatureBaseline')

        return results[0]


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
    #task_id = openMLname2task['iris']
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
    #task_id = openMLname2task['kc2']

    dataset = None

    all_results: List[CandidateFeature] = []
    for rotation in range(10):
        selector = Run_RawFeatures(dataset, reader=OnlineOpenMLReader(task_id, 1, rotation), score=make_scorer(roc_auc_score)) #make_scorer(f1_score, average='micro') #make_scorer(roc_auc_score)
        #selector = Run_RawFeatures(dataset, score=make_scorer(roc_auc_score))
        #selector = ExploreKitSelection(dataset, KNeighborsClassifier(), {'n_neighbors': np.arange(3,10), 'weights': ['uniform','distance'], 'metric': ['minkowski','euclidean','manhattan']})

        all_results.append(selector.run())


    print("Average test score: " + str(np.mean([c.runtime_properties['test_score'] for c in all_results])))







