from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack
from sklearn.metrics import accuracy_score
from library.initialisation.InitialTraining import InitialTraining
import time

import numpy as np
import matplotlib.pyplot as plt

from library.stat_model.SVC_Model import SVC_Model
from library.Data.News import News




def train_model(X_train, y_train, stat_model, trainsize, traintime):
    #stat_model.optimize_hyperparameters(X_train, y_train, folds=5)
    # params = stat_model.best_params

    params = {'kernel': 'rbf', 'C': 10, 'probability': True, 'gamma': 0.1}
    #params = {'kernel': 'rbf', 'C': 1, 'probability': True, 'gamma': 0.001} #with scale
    print params

    train_start_time = time.time()
    stat_model.train(X_train, y_train, params)
    traintime.append((time.time() - train_start_time))
    trainsize.append(X_train.shape[0])




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


random_seed = 42

#print newsgroups_train

analyzer="word"
ngrams = 1
pipeline = Pipeline([('vect', CountVectorizer(analyzer=analyzer,
                                              lowercase=False,
                                              ngram_range=(1, ngrams))),
                     ('tfidf', TfidfTransformer()),
                     #('scale', StandardScaler(with_mean=False))
                    ])

pipeline.fit(data_train_x)
x = pipeline.transform(data_train_x)
y = data_train_y

X_test = pipeline.transform(data_test_x)
y_test = data_test_y


all_list = []
time_list_all = []

#incremental training

for run in range(1):

    stat_model = SVC_Model()

    random_seed += 1

    my_random = np.random.RandomState(random_seed)

    X_train = None
    y_train = []

    accuracy_list = [0]

    #create sample for active learning
    ids = np.arange(x.shape[0])
    my_random.shuffle(ids)

    samplesize = int(x.shape[0] *0.1)

    sample_x = x[ids[0:samplesize]]
    sample_y = y[ids[0:samplesize]]

    time_list = []

    start_time = time.time()

    trainsize = []
    traintime = []
    predicttime = []
    predictsize = []


    ids_set = set()

    # create in initial training set
    init = InitialTraining(x, y, n=2, random_seed=random_seed)
    X_train, y_train = init.generate()

    # first model

    train_model(X_train, y_train, stat_model,trainsize, traintime)
    test_model(stat_model, X_test, y_test, accuracy_list)

    # start active learning
    for t in range(200):
        #predict for all records in sample
        proba_sample = stat_model.model.predict_proba(sample_x)

        '''
        for predict_i in range(1, x.shape[0], 10):
            sample_x = x[ids[0:predict_i]]
            sample_y = y[ids[0:predict_i]]


            print predict_i
            predict_start_time = time.time()
            proba_sample = stat_model.model.predict_proba(sample_x)
            predicttime.append((time.time() - predict_start_time))
            predictsize.append(sample_x.shape[0])
        break
        '''

        #select least certain record:
        topk = np.argsort(proba_sample, axis=1)[:,proba_sample.shape[1]-2:proba_sample.shape[1]]
        max_uncertainty = -1
        record_id = -1
        for s_i in range(proba_sample.shape[0]):
            if not s_i in ids_set:
                uncertainty = 1.0 - np.sum(proba_sample[s_i, topk[s_i]])
                #print uncertainty
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    record_id = s_i

        ids_set.add(record_id)
        X_train = vstack((X_train, x[record_id]))
        y_train.append(y[record_id])

        print "time: " + str(t)

        train_model(X_train, y_train, stat_model,trainsize, traintime)
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


fig, ax = plt.subplots()
ax.plot(range(len(trainsize)), traintime, label='training time')
plt.show()

print list(trainsize)
print list(traintime)


print list(predicttime)
print list(predictsize)