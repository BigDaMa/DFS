from sklearn.preprocessing import LabelEncoder
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
import xgboost as xgb
import time


class ActiveLearner:
    def __init__(self,
                 datasets: List[Reader],
                 all_feature_candidates: List[CandidateFeature],
                 working_candidates_ids: List[int],
                 working_candidate_id_to_dataset_id: List[int],
                 metadata_feature_matrix,
                 feature_names,
                 runtime: int,
                 batch_size: int =10,
                 classifier=LogisticRegression()):

        self.datasets = datasets
        self.all_feature_candidates = all_feature_candidates
        self.working_candidates_ids = working_candidates_ids
        self.working_candidate_id_to_dataset_id = working_candidate_id_to_dataset_id
        self.metadata_feature_matrix = metadata_feature_matrix
        self.feature_names = feature_names

        self.runtimeMinutes = runtime
        self.batch_size = batch_size
        self.classifier = classifier

        self.xGBoostMatrix = xgb.DMatrix(metadata_feature_matrix, feature_names=self.feature_names)


    def create_initial_training_set(self, N=50):
        #sample N random candidates
        arr = np.arange(self.metadata_feature_matrix.shape[0])
        np.random.shuffle(arr)

        y = []
        selected_ids = []

        #randomly sample a new instance until we have both classes in the training
        while not (np.sum(np.array(y)) >= 5 and np.sum(np.array(y)) <= len(y) - 5 and len(selected_ids) >=N):
            new_selected_id = arr[0]
            selected_ids.append(new_selected_id)
            arr = arr[1:len(arr)]
            y.append(self.create_label_by_running_classifier(new_selected_id))

        return y, selected_ids


    def train(self, X, y):
        model = XGBoostClassifier()
        model.hyperparameter_optimization(X, y, self.feature_names)
        return model


    def run(self):
        start = time.time()
        y, selected_ids = self.create_initial_training_set()

        print("Selected ids: " + str(len(selected_ids)))

        while time.time() - start <= 60 * self.runtimeMinutes:
            X = self.metadata_feature_matrix[selected_ids, :]
            self.model = self.train(X, y)
            least_certain_tuple_ids = self.model.get_k_least_certain_tuples(self.xGBoostMatrix, k=10)
            #label selected tuples:
            for tuple_i in least_certain_tuple_ids:
                y.append(self.create_label_by_running_classifier(tuple_i))
                selected_ids.append(tuple_i)


    def create_label_by_running_classifier(self, working_candidate_id, folds=10):
        try:
            # run plain attribute
            # first hstack parent features
            parent_feature_list = []
            for p in self.all_feature_candidates[self.working_candidates_ids[working_candidate_id]].parents:
                parent_feature_list.append(p.materialize()['train'].reshape(-1, 1))

            current_plain_matrix = np.hstack(parent_feature_list)

            current_target = self.datasets[self.working_candidate_id_to_dataset_id[working_candidate_id]].splitted_target['train']
            current_target = LabelEncoder().fit_transform(current_target)

            cv_results = cross_validate(self.classifier,
                                        current_plain_matrix,
                                        current_target,
                                        cv=folds,
                                        scoring=make_scorer(roc_auc_score, average='micro'),
                                        return_train_score=False)
            raw_score = np.average(cv_results['test_score'])

            materialized_feature = self.all_feature_candidates[self.working_candidates_ids[working_candidate_id]].materialize()['train'].reshape(-1, 1)

            # run transformed attribute
            # think about hyperparameter tuning

            cv_results = cross_validate(self.classifier,
                                        materialized_feature,
                                        current_target,
                                        cv=folds,
                                        scoring=make_scorer(roc_auc_score, average='micro'),
                                        return_train_score=False)

            transformed_score = np.average(cv_results['test_score'])

            return transformed_score > raw_score
        except:
            return False