import optuna
from sklearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.CategoricalMissingTransformer import CategoricalMissingTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import time
import resource
import copy
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.automl_parameters as mp_global
import multiprocessing
from pathlib import Path
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.define_space as myspace
import pickle
import os

class TimeException(Exception):
    def __init__(self, message="Time is over!"):
        self.message = message
        super().__init__(self.message)


def evaluatePipeline(key, return_dict):
    try:
        balanced = mp_global.mp_store[key]['balanced']
        p = mp_global.mp_store[key]['p']
        number_of_cvs = mp_global.mp_store[key]['number_of_cvs']
        cv = mp_global.mp_store[key]['cv']
        scorer = mp_global.mp_store[key]['scorer']
        X = mp_global.mp_store[key]['X']
        y = mp_global.mp_store[key]['y']
        main_memory_budget_gb = mp_global.mp_store[key]['main_memory_budget_gb']

        size = int(main_memory_budget_gb * 1024.0 * 1024.0 * 1024.0)
        resource.setrlimit(resource.RLIMIT_AS, (size, resource.RLIM_INFINITY))

        start_training = time.time()

        if balanced:
            p.fit(X, y, classifier__sample_weight=compute_sample_weight(class_weight='balanced', y=y))
        else:
            p.fit(X, y)
        training_time = time.time() - start_training

        return_dict[key + 'pipeline'] = copy.deepcopy(p)
        #pickle instead otherwise get a segmentation error

        with open('/tmp/my_pipeline' + str(key) + '.p', "wb") as pickle_pipeline_file:
            pickle.dump(p, pickle_pipeline_file)



        scores = []
        for cv_num in range(number_of_cvs):
            my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time())).split(X, y)
            #my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(42)).split(X, y)
            for train_ids, test_ids in my_splits:
                if balanced:
                    p.fit(X[train_ids, :], y[train_ids], classifier__sample_weight=compute_sample_weight(class_weight='balanced', y=y[train_ids]))
                else:
                    p.fit(X[train_ids, :], y[train_ids])
                scores.append(scorer(p, X[test_ids, :], pd.DataFrame(y[test_ids])))

        return_dict[key + 'result'] = np.mean(scores)
    except Exception as e:
        print(p)
        print(str(e) + '\n\n')





class MyAutoML:
    def __init__(self, cv=5, number_of_cvs=1, evaluation_budget=np.inf, time_search_budget=10*60, n_jobs=1, space=None, study=None, main_memory_budget_gb=4):
        self.cv = cv
        self.time_search_budget = time_search_budget
        self.n_jobs = n_jobs
        self.evaluation_budget = evaluation_budget
        self.number_of_cvs = number_of_cvs

        self.classifier_list = myspace.classifier_list
        self.preprocessor_list = myspace.preprocessor_list
        self.scaling_list = myspace.scaling_list
        self.categorical_encoding_list = myspace.categorical_encoding_list

        #generate binary or mapping for each hyperparameter


        self.space = space
        self.study = study
        self.main_memory_budget_gb = main_memory_budget_gb

        #print("number of hyperparameters: " + str(len(self.space.parameters_used)))

        #signal.signal(signal.SIGSEGV, signal_handler)


    def get_best_pipeline(self):
        try:
            return self.study.best_trial.user_attrs['pipeline']
        except:
            return None


    def predict(self, X):
        best_pipeline = self.get_best_pipeline()
        return best_pipeline.predict(X)


    def fit(self, X, y, sample_weight=None, categorical_indicator=None, scorer=None):
        self.start_fitting = time.time()

        def objective1(trial):
            start_total = time.time()

            try:

                self.space.trial = trial

                imputer = SimpleImputerOptuna()
                imputer.init_hyperparameters(self.space, X, y)

                scaler = self.space.suggest_categorical('scaler', self.scaling_list)
                scaler.init_hyperparameters(self.space, X, y)

                onehot_transformer = self.space.suggest_categorical('categorical_encoding', self.categorical_encoding_list)
                onehot_transformer.init_hyperparameters(self.space, X, y)

                preprocessor = self.space.suggest_categorical('preprocessor', self.preprocessor_list)
                preprocessor.init_hyperparameters(self.space, X, y)

                classifier = self.space.suggest_categorical('classifier', self.classifier_list)
                classifier.init_hyperparameters(self.space, X, y)

                balanced = False
                if isinstance(classifier, KNeighborsClassifierOptuna) or \
                        isinstance(classifier, QuadraticDiscriminantAnalysisOptuna) or \
                        isinstance(classifier, PassiveAggressiveOptuna) or \
                        isinstance(classifier, HistGradientBoostingClassifierOptuna):
                    balanced = False
                else:
                    balanced = self.space.suggest_categorical('balanced', [True, False])

                numeric_transformer = Pipeline([('imputation', imputer), ('scaler', scaler)])
                categorical_transformer = Pipeline([('removeNAN', CategoricalMissingTransformer()), ('onehot_transform', onehot_transformer)])


                my_transformers = []
                if np.sum(np.invert(categorical_indicator)) > 0:
                    my_transformers.append(('num', numeric_transformer, np.invert(categorical_indicator)))

                if np.sum(categorical_indicator) > 0:
                    my_transformers.append(('cat', categorical_transformer, categorical_indicator))


                data_preprocessor = ColumnTransformer(transformers=my_transformers)

                my_pipeline = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor),
                              ('classifier', classifier)])

                key = 'My_processs' + str(time.time()) + "##" + str(np.random.randint(0,1000))

                mp_global.mp_store[key] = {}

                mp_global.mp_store[key]['balanced'] = balanced
                mp_global.mp_store[key]['p'] = copy.deepcopy(my_pipeline)
                mp_global.mp_store[key]['number_of_cvs'] = self.number_of_cvs
                mp_global.mp_store[key]['cv'] = self.cv
                mp_global.mp_store[key]['scorer'] = scorer
                mp_global.mp_store[key]['X'] = X
                mp_global.mp_store[key]['y'] = y
                mp_global.mp_store[key]['main_memory_budget_gb'] = self.main_memory_budget_gb

                already_used_time = time.time() - self.start_fitting

                if already_used_time + 2 >= self.time_search_budget:  # already over budget
                    time.sleep(2)
                    return -np.inf

                remaining_time = np.min([self.evaluation_budget, self.time_search_budget - already_used_time])

                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                my_process = multiprocessing.Process(target=evaluatePipeline, name='start'+key, args=(key, return_dict,))
                my_process.start()

                my_process.join(int(remaining_time))

                # If thread is active
                while my_process.is_alive():
                    # Terminate foo
                    my_process.terminate()
                    my_process.join()

                del mp_global.mp_store[key]

                result = -np.inf
                if key + 'result' in return_dict:
                    result = return_dict[key + 'result']

                trial.set_user_attr('total_time', time.time() - start_total)

                try:
                    if self.study.best_value < result:
                        if Path('/tmp/my_pipeline' + str(key) + '.p').is_file():
                            with open('/tmp/my_pipeline' + str(key) + '.p', "rb") as pickle_pipeline_file:
                                trial.set_user_attr('pipeline', pickle.load(pickle_pipeline_file))
                            os.remove('/tmp/my_pipeline' + str(key) + '.p')
                except:
                    if Path('/tmp/my_pipeline' + str(key) + '.p').is_file():
                        with open('/tmp/my_pipeline' + str(key) + '.p', "rb") as pickle_pipeline_file:
                            trial.set_user_attr('pipeline', pickle.load(pickle_pipeline_file))
                        os.remove('/tmp/my_pipeline' + str(key) + '.p')

                return result
            except Exception as e:
                print(str(e) + '\n\n')
                return -np.inf

        if type(self.study) == type(None):
            self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective1, timeout=self.time_search_budget, n_jobs=self.n_jobs, catch=(TimeException,))
        return self.study.best_value




if __name__ == "__main__":
    auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

    '''
    my_openml_datasets = [  # 31, #German Credit # yes
        # 1464,  # Blood Transfusion
        # 333,  # Monks Problem 1
        334,  # Monks Problem 2
        50,  # TicTacToe
        # 1504,  # steel plates fault #yes
        # 3,  # kr-vs-kp #works
        # 1494,  # qsar-biodeg #yes
        # 1510,  # wdbc #yes
        # 1489,  # phoneme #yes
        1590 #adult #yes
        1067, #kc1
        37, #diabetes
        1487, #ozon
        1479, #hill valley
        1063, # kc2
        1471, #eeg
        1467, #climatemodel
        44, #spambase
        1461, #bankmarketing
        4135, #amazon
    ]
    '''

    dataset = openml.datasets.get_dataset(31)

    #dataset = openml.datasets.get_dataset(31)
    #dataset = openml.datasets.get_dataset(1590)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    print(X)


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1, stratify=y, train_size=0.6)

    gen = SpaceGenerator()
    space = gen.generate_params()

    from anytree import RenderTree

    for pre, _, node in RenderTree(space.parameter_tree):
        print("%s%s: %s" % (pre, node.name, node.status))

    search = MyAutoML(cv=5, n_jobs=1, time_search_budget=500, space=space)

    begin = time.time()

    best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

    test_score = auc(search.get_best_pipeline(), X_test, y_test)

    print("result: " + str(best_result) + " test: " + str(test_score))

    print(time.time() - begin)