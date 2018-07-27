from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from library.cluster_selection.UCB1_ZOMBIE import UCB1
from sets import Set
from sklearn.metrics import accuracy_score
import time
from library.Data.News import News
from library.feature.TFIDF import TFIDF

from library.stat_model.SVC_Model import SVC_Model
from library.stat_model.NaiveBayes import NaiveBayes
from sacred import Experiment
from sacred.observers import MongoObserver


def train_model(X_train, y_train, stat_model):
    #stat_model.optimize_hyperparameters(X_train, y_train, folds=5)
    # params = stat_model.best_params

    #params = {'kernel': 'rbf', 'C': 10000, 'probability': True, 'gamma': 0.1}
    params = {'kernel': 'linear', 'C': 100, 'probability': True, 'class_weight': 'balanced'}
    #params = {'kernel': 'rbf', 'C': 1, 'probability': True, 'gamma': 0.001} #with scale
    #params = {'alpha': alpha}
    #print params
    #params = {'alpha': 0.1}

    stat_model.train(X_train, y_train, params)




def test_model(stat_model, X_test, y_test, fscore_list):
    y_true, y_pred = y_test, stat_model.predict(X_test)

    fscore_list.append(accuracy_score(y_true, y_pred))
    print(fscore_list[-1])

# load data
# data
data = News()
#data = Spam()
data_train_x, data_train_y = data.get_train()
data_test_x, data_test_y = data.get_test()

#print newsgroups_train

featurizer = TFIDF()
(x, y, X_test, y_test) = featurizer.featurize(data_train_x, data_train_y, data_test_x, data_test_y)
#incremental training

stat_model = SVC_Model()
#stat_model = NaiveBayes()

X_train = None
y_train = []

accuracy_list = [0]



#print x.shape[0]

# first model
import numpy as np
from scipy.sparse import vstack

ids = np.arange(x.shape[0])
np.random.shuffle(ids)

X_train = None
y_train = []

length = 2000#x.shape[0]

if length < x.shape[0]:
    for id in ids[0:length]:
        if X_train == None:
            X_train = x[id]
        else:
            X_train = vstack((X_train, x[id]))
        y_train.append(y[id])
else:
    X_train = x
    y_train = y


print X_train.shape[0]

start_time = time.time()
train_model(X_train, y_train, stat_model)
test_model(stat_model, X_test, y_test, accuracy_list)


support_vector_ids = stat_model.model.support_

print "number support vectors: " + str(len(support_vector_ids))
