from hyperopt import hp
from hpsklearn import HyperoptEstimator
from hpsklearn.components import any_classifier
from hpsklearn.components import any_preprocessing
from hyperopt import tpe
from sklearn.pipeline import Pipeline

import hyperopt.pyll.stochastic

import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
from sklearn.model_selection import cross_val_score
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

print(hyperopt.__version__)

os.environ['OMP_NUM_THREADS'] = '1'


auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

dataset = openml.datasets.get_dataset(31)
#dataset = openml.datasets.get_dataset(1590)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

for constraint in [1,5]:

    def objective(x):
        start_time = time.time()

        onehot_enc = ColumnTransformer(
            [('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32), np.nonzero(categorical_indicator)[0])], remainder='passthrough')

        classifier = x['clf']
        p = None
        try:
            preprocessor = x['preprocessor'][0]
            p = Pipeline([('imputation', SimpleImputer(missing_values=np.nan, strategy='median')), ('cats', onehot_enc),  ('preprocessing', preprocessor), ('classifier', classifier)])
        except:
            p = Pipeline([('imputation', SimpleImputer(missing_values=np.nan, strategy='median')), ('cats', onehot_enc), ('classifier', classifier)])

        try:
            start_training = time.time()
            p.fit(X_train, y_train)
            training_time = time.time() - start_training

            if training_time > constraint:
                return {'loss': np.inf, 'status': STATUS_OK, 'training_time': training_time,
                        'total_time': time.time() - start_time}

            scores = cross_val_score(p, X_train, y_train, cv=5, scoring=auc, n_jobs=5)

            total_time = time.time() - start_time
            return {'loss': -1 * np.mean(scores), 'status': STATUS_OK, 'training_time': training_time,
                    'total_time': total_time}
        except:
            total_time = time.time() - start_time
            return {'loss': np.inf, 'status': STATUS_OK, 'training_time': 0, 'total_time': total_time}


    pipeline_space = {'clf': any_classifier('my_clf'), 'preprocessor': any_preprocessing('my_prep')}

    print(pipeline_space)

    trials = Trials()
    best = fmin(objective,
        space=pipeline_space,
        algo=tpe.suggest,
        max_evals=200, trials=trials)


    print(trials.best_trial)


    pickle.dump(trials, open("/tmp/trials.p", "wb"))

