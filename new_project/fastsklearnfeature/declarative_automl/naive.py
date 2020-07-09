from hyperopt import hp
from hpsklearn import HyperoptEstimator
from hpsklearn.components import any_classifier
from hpsklearn.components import any_preprocessing
from hyperopt import tpe
from sklearn.pipeline import Pipeline

import hyperopt.pyll.stochastic




def create_random_pipeline():

    pipeline_space = {'clf': any_classifier('my_clf'), 'preprocessor': any_preprocessing('my_prep')}

    sample = hyperopt.pyll.stochastic.sample(pipeline_space)

    classifier = sample['clf']
    p = None
    try:
        preprocessor = sample['preprocessor'][0]
        p = Pipeline([('preprocessing', preprocessor), ('classifier', classifier)])
    except:
        p = Pipeline([('classifier', classifier)])



    return p

p = create_random_pipeline()

print(p)