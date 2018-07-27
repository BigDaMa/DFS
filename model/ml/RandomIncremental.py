from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack
from library.initialisation.InitialTraining import InitialTraining
import time

import numpy as np
import matplotlib.pyplot as plt

from library.stat_model.NaiveBayes import NaiveBayes
from library.Data.News import News
from library.feature.TFIDF import TFIDF



def train_model(X_train, y_train, stat_model, traintime):
    #stat_model.optimize_hyperparameters(X_train, y_train, folds=5)
    # params = stat_model.best_params

    params = {'alpha': 0.1}
    print params

    train_start_time = time.time()
    stat_model.partial_train(X_train, y_train, params)
    traintime.append((time.time() - train_start_time))




def test_model(stat_model, X_test, y_test, fscore_list):
    y_true, y_pred = y_test, stat_model.predict(X_test)

    predictive_accuracy = np.mean(y_pred == y_true)
    fscore_list.append(predictive_accuracy)
    print(predictive_accuracy)

# load data
# data
data = News()
#data = Spam()
data_train_x, data_train_y = data.get_train()
data_test_x, data_test_y = data.get_test()


random_seed = 42

#print newsgroups_train

featurizer = TFIDF()
(x, y, X_test, y_test) = featurizer.featurize(data_train_x, data_train_y, data_test_x, data_test_y)

all_list = []
time_list_all = []

#incremental training

for run in range(1):

    stat_model = NaiveBayes()

    random_seed += 1

    my_random = np.random.RandomState(random_seed)

    X_train = None
    y_train = []

    accuracy_list = [0]

    #create sample for active learning
    ids = np.arange(x.shape[0])
    my_random.shuffle(ids)

    samplesize = int(x.shape[0]) #*0.1

    sample_x = x[ids[0:samplesize]]
    sample_y = y[ids[0:samplesize]]

    time_list = []

    traintime = []

    start_time = time.time()


    ids_set = set()


    id_list_rand = np.arange(sample_x.shape[0])
    np.random.shuffle(id_list_rand)

    # create in initial training set
    init = InitialTraining(x, y, n=2, random_seed=random_seed)
    X_train, y_train = init.generate()

    print "init: " + str(X_train.shape[0])

    # first model

    train_model(X_train, y_train, stat_model, [])
    test_model(stat_model, X_test, y_test, accuracy_list)

    # start active learning
    for t in range(1000):
        while True:
            record_id = id_list_rand[np.random.randint(0, sample_x.shape[0])]
            if not record_id in ids_set:
                break

        ids_set.add(record_id)

        print "time: " + str(t)

        print type(x[record_id])

        train_model(x[record_id], [y[record_id]], stat_model, traintime)
        test_model(stat_model, X_test, y_test, accuracy_list)

        time_list.append((time.time() - start_time))

    all_list.append(accuracy_list)
    time_list_all.append(time_list)


mean_list = list(np.mean(all_list, axis=0))
mean_time = list(np.mean(time_list_all, axis=0))

fig, ax = plt.subplots()
ax.plot(range(len(mean_list)), mean_list, label='Accuracy')
plt.show()

print list(mean_list)
print list(mean_time)

print list(traintime)