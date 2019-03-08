from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.transformations.Transformation import Transformation
from typing import List
import numpy as np
from fastsklearnfeature.reader.Reader import Reader
from fastsklearnfeature.splitting.Splitter import Splitter
import time
from fastsklearnfeature.candidate_generation.explorekit.Generator import Generator
from fastsklearnfeature.candidates.RawFeature import RawFeature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from fastsklearnfeature.configuration.Config import Config
from sklearn.pipeline import FeatureUnion
import itertools
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from autofeat import AutoFeatRegression
import joblib
import pandas as pd
import featuretools as ft

class SissoExperiment:
    def __init__(self, dataset_config, classifier=LogisticRegression(), grid_search_parameters={'classifier__penalty': ['l2'],
                                                                                                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                                                                'classifier__solver': ['lbfgs']}):
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
        self.candidates = g.generate_all_candidates()
        print("Number candidates: " + str(len(self.candidates)))

    def generate_target(self):
        current_target = self.dataset.splitted_target['train']
        self.current_target = LabelEncoder().fit_transform(current_target)

    def evaluate(self, candidate, score=make_scorer(roc_auc_score, average='micro'), folds=10):
        parameters = self.grid_search_parameters


        '''
        if not isinstance(candidate, CandidateFeature):
            pipeline = Pipeline([('features',FeatureUnion(

                        [(p.get_name(), p.pipeline) for p in candidate]
                    )),
                ('classifier', self.classifier)
            ])
        else:
            pipeline = Pipeline([('features', FeatureUnion(
                [
                    (candidate.get_name(), candidate.pipeline)
                ])),
                 ('classifier', self.classifier)
                 ])
        '''

        result = {}

        ''''
        clf = GridSearchCV(pipeline, parameters, cv=self.preprocessed_folds, scoring=score, iid=False, error_score='raise')
        clf.fit(self.dataset.splitted_values['train'], self.current_target)
        result['score'] = clf.best_score_
        result['hyperparameters'] = clf.best_params_
        '''

        feateng_cols = ['age', 'sex', 'chest', 'resting_blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
                        'resting_electrocardiographic_results', 'maximum_heart_rate_achieved',
                        'exercise_induced_angina', 'oldpeak', 'slope', 'number_of_major_vessels', 'thal']


        print(self.current_target)

        '''
        df = pd.DataFrame(data=self.dataset.splitted_values['train'], columns=feateng_cols)
        df['id'] = pd.Series(range(len(df)), index=df.index)

        df_target = pd.DataFrame(data=self.current_target, columns=['target'])
        df_target['id'] = pd.Series(range(len(df)), index=df_target.index)

        print(df)


        es = ft.EntitySet("example")
        es.entity_from_dataframe(dataframe=df, entity_id="heart",index="id")
        es.entity_from_dataframe(dataframe=df_target, entity_id="target", index="id")

        new_relationship = ft.Relationship(es["heart"]["id"], es["target"]["id"])
        es = es.add_relationship(new_relationship)

        
        '''

        df = pd.DataFrame(data=self.dataset.splitted_values['train'], columns=feateng_cols)
        df['id'] = pd.Series(range(len(df)), index=df.index)
        df['target'] = pd.Series(self.current_target, index=df.index).map({0: 'healthy', 1: 'ill'})

        es = ft.EntitySet("example")
        es.entity_from_dataframe(dataframe=df, entity_id="heart", index="id")


        es.normalize_entity(base_entity_id='heart', new_entity_id='target_e', index='id',
                            additional_variables=['target'])

        print(es)

        feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='target_e', max_depth=6, verbose=1,
                                              n_jobs=4)

        df = es['heart'].df
        dfs_transformer = ft.wrappers.DFSTransformer(entityset=es, target_entity='target_e', max_features=3)

        #does not work
        dfs_transformer.fit(df['id'])
        transformed = dfs_transformer.transform([0])
        print(transformed)


        drop_cols = []
        for col in feature_matrix:
            if col == 'target':
                pass
            else:
                if 'target' in col:
                    drop_cols.append(col)

        feature_matrix = feature_matrix[[x for x in feature_matrix if x not in drop_cols]]
        print(feature_matrix.columns)




        return result



    '''
    def evaluate_candidates(self, candidates):
        self.preprocessed_folds = []
        for train, test in StratifiedKFold(n_splits=10, random_state=42).split(self.dataset.splitted_values['train'], self.current_target):
            self.preprocessed_folds.append((train, test))

        pool = mp.Pool(processes=int(Config.get("parallelism")))
        results = pool.map(self.evaluate_single_candidate, candidates)
        return results

    '''
    def evaluate_candidates(self, candidates):
        self.preprocessed_folds = []
        for train, test in StratifiedKFold(n_splits=10, random_state=42).split(self.dataset.splitted_values['train'],
                                                                               self.current_target):
            self.preprocessed_folds.append((train, test))

        results = []
        for c in candidates:
            results.append(self.evaluate_single_candidate(c))
        return results



    '''
    def evaluate_single_candidate(self, candidate):
        result = {}
        time_start_gs = time.time()
        try:
            result = self.evaluate(candidate)
            #print("feature: " + str(candidate) + " -> " + str(new_score))
        except Exception as e:
            print(str(candidate) + " -> " + str(e))
            result['score'] = -1.0
            result['hyperparameters'] = {}
            pass
        result['candidate'] = candidate
        result['time'] = time.time() - time_start_gs
        return result


    '''
    def evaluate_single_candidate(self, candidate):
        new_score = -1.0
        new_score = self.evaluate(candidate)
        return new_score



    def run(self):
        # generate all candidates
        self.generate()
        #starting_feature_matrix = self.create_starting_features()
        self.generate_target()

        print([r.name for r in self.dataset.raw_features])


        plain_attributes = CandidateFeature(IdentityTransformation(len(self.dataset.raw_features)), self.dataset.raw_features)


        self.evaluate_candidates([plain_attributes])




#statlog_heart.csv=/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv
#statlog_heart.target=13

if __name__ == '__main__':
    dataset = (Config.get('statlog_heart.csv'), int(Config.get('statlog_heart.target')))

    selector = SissoExperiment(dataset)

    selector.run()






