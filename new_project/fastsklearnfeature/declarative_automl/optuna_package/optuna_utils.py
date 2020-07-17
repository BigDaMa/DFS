import time
import numpy as np

def id_name(name):
    #return name + str(time.time()) + str(np.random.randint(0, 1000)) + '_'
    return name

def categorical(trial, name, categories):
    list_id = trial.suggest_categorical(id_name(name), list(range(len(categories))))
    return categories[list_id]

