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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from fastfeature.plotting.plotter import cool_plotting
import pickle
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

class ExploreKitSelection:
    def __init__(self, dataset_config, classifier=LogisticRegression(), grid_search_parameters={'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}):
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
        self.candidates = g.generate_candidates()
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



    '''
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



    def evaluate(self, feature_matrix, feature_names, score=make_scorer(roc_auc_score, average='micro'), folds=10):
        cv_params = {'min_child_weight': [1, 3, 5],
                     'subsample': [0.7, 0.8, 0.9],
                     'max_depth': [3, 5, 7],
                     'n_estimators': [1, 10, 100, 1000]
                     }

        ind_params = {  # 'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
            'learning_rate': 0.1,  # we could optimize this: 'learning_rate': [0.1, 0.01]
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 0,
            'objective': 'binary:logistic',
            'n_jobs': 4
        }

        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='f1', cv=folds, n_jobs=1)

        optimized_GBM.fit(feature_matrix, self.current_target)

        print("all_results: " + str(optimized_GBM.cv_results_['mean_test_score']))
        print("best config: " + str(optimized_GBM.best_params_))
        print("best score: " + str(optimized_GBM.best_score_))

        print(str(optimized_GBM.best_estimator_.get_booster().get_score(importance_type='total_gain')))

        for feature_i in range(len(feature_names)):
            print("f" + str(feature_i) + ": " + feature_names[feature_i])



        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        xgdmat = xgb.DMatrix(feature_matrix, self.current_target, feature_names=feature_names)
        model = xgb.train(our_params, xgdmat, verbose_eval=False)

        print(model.get_score(importance_type='total_gain'))





    def create_starting_features(self):
        Fi: List[RawFeature]= self.dataset.raw_features

        #materialize and numpyfy the features
        starting_feature_matrix = np.zeros((Fi[0].materialize()['train'].shape[0], len(Fi)))
        feature_names = []
        for f_index in range(len(Fi)):
            starting_feature_matrix[:, f_index] = Fi[f_index].materialize()['train']
            feature_names.append(Fi[f_index].get_name())
        return starting_feature_matrix, feature_names

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


    def get_traceability(self, candidate_id):
        return self.traceability[candidate_id]

    def get_interpretability(self, candidate_id):
        return 1.0 - ((self.candidate_id_to_ranked_id[candidate_id] + 1) / float(len(self.candidate_id_to_ranked_id)))



    def run(self):
        # generate all candidates
        self.generate()
        starting_feature_matrix, feature_names = self.create_starting_features()
        self.generate_target()

        self.evaluate(starting_feature_matrix, feature_names)

    def plot_accuracy_vs_interpretability(self, start_score, new_scores, ids):
        interpretability = []
        names = []

        for i in range(len(ids)):
            interpretability.append(self.get_interpretability(ids[i]))
            cand = self.candidates[ids[i]]
            names.append(str(cand) + ": d: " + str(cand.get_transformation_depth()) + "& t:" + str(cand.get_number_of_transformations()))

        all_data = {}
        all_data['start_score'] = start_score
        all_data['new_scores'] = new_scores
        all_data['names'] = names
        all_data['ids'] = ids
        all_data['interpretability'] = interpretability

        pickle.dump(all_data, open("/tmp/chart.p", "wb"))

        cool_plotting(interpretability, new_scores, names, start_score)





if __name__ == '__main__':
    dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13)
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

    start_score, new_scores, ids = selector.run()

    selector.plot_accuracy_vs_interpretability(start_score, new_scores, ids)



